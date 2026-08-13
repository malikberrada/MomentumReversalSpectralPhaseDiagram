from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import mrspd_native as _mrspd_native
except ImportError:
    _mrspd_native = None


@dataclass(frozen=True)
class Config:
    start: str = "2005-01-01"
    end: str | None = None
    horizons: Tuple[int, ...] = (5, 10, 21, 42, 63, 126)
    spectral_spans: Tuple[int, ...] = (5, 10, 21, 42, 63, 126)
    vol_span: int = 33
    demean_window: int = 252
    spectral_window: int = 756
    beta_window: int = 252
    anchor_stride: int = 5
    z_clip: float = 8.0
    min_history_frac: float = 0.67
    n_walkforward_splits: int = 4
    min_train_frac: float = 0.50
    download_batch_size: int = 40
    retries: int = 3
    market_ticker: str = "SPY"
    include_raw_mode: bool = True
    include_market_residual_mode: bool = True
    random_seed: int = 20260812
    native_backend: str = "auto"
    native_batch_size: int = 128


def read_universe(path: Path) -> pd.DataFrame:
    u = pd.read_csv(path)
    required = {"ticker", "asset_class"}
    if not required.issubset(u.columns):
        raise ValueError(f"Universe must contain columns {sorted(required)}")
    u = u.loc[:, ["ticker", "asset_class"]].dropna().copy()
    u["ticker"] = u["ticker"].astype(str).str.strip().str.upper()
    u["asset_class"] = u["asset_class"].astype(str).str.strip()
    u = u.drop_duplicates("ticker").reset_index(drop=True)
    return u


def _normalize_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).title() for c in out.columns]
    wanted = ["Open", "High", "Low", "Close", "Volume"]
    for c in wanted:
        if c not in out.columns:
            out[c] = np.nan
    out = out[wanted]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _extract_batch(raw: pd.DataFrame, tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
    ans: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return ans
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(map(str, raw.columns.get_level_values(0)))
        lv1 = set(map(str, raw.columns.get_level_values(1)))
        for t in tickers:
            try:
                if t in lv0:
                    d = raw[t]
                elif t in lv1:
                    d = raw.xs(t, axis=1, level=1)
                else:
                    continue
                d = _normalize_one_ticker(d)
                if d["Close"].notna().sum() >= 100:
                    ans[t] = d
            except Exception:
                continue
    else:
        if len(tickers) == 1:
            d = _normalize_one_ticker(raw)
            if d["Close"].notna().sum() >= 100:
                ans[tickers[0]] = d
    return ans


def download_ohlcv(universe: pd.DataFrame, cache_path: Path, cfg: Config, force: bool = False) -> Dict[str, pd.DataFrame]:
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    import yfinance as yf

    tickers = universe["ticker"].tolist()
    if cfg.market_ticker not in tickers:
        tickers.append(cfg.market_ticker)

    result: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), cfg.download_batch_size):
        batch = tickers[i : i + cfg.download_batch_size]
        got: Dict[str, pd.DataFrame] = {}
        for attempt in range(cfg.retries):
            try:
                raw = yf.download(
                    tickers=batch,
                    start=cfg.start,
                    end=cfg.end,
                    interval="1d",
                    auto_adjust=True,
                    repair=True,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    timeout=30,
                    multi_level_index=True,
                )
                got.update(_extract_batch(raw, batch))
                missing = [t for t in batch if t not in got]
                if not missing:
                    break
            except Exception as e:
                print(f"batch {i//cfg.download_batch_size+1} attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))

        missing = [t for t in batch if t not in got]
        for t in missing:
            for attempt in range(cfg.retries):
                try:
                    raw = yf.download(
                        tickers=t,
                        start=cfg.start,
                        end=cfg.end,
                        interval="1d",
                        auto_adjust=True,
                        repair=True,
                        actions=False,
                        threads=False,
                        progress=False,
                        timeout=30,
                        multi_level_index=False,
                    )
                    one = _extract_batch(raw, [t])
                    if t in one:
                        got[t] = one[t]
                        break
                except Exception as e:
                    print(f"retry {t} attempt {attempt+1}: {e}", file=sys.stderr)
                time.sleep(1.5 * (attempt + 1))

        result.update(got)
        print(f"downloaded {len(result)}/{len(tickers)} tickers")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    failed = sorted(set(tickers) - set(result))
    if failed:
        print("WARNING failed tickers:", ", ".join(failed), file=sys.stderr)
    return result


def _rolling_beta(asset_r: pd.Series, market_r: pd.Series, window: int) -> pd.Series:
    pair = pd.concat([asset_r.rename("a"), market_r.rename("m")], axis=1)
    cov = pair["a"].rolling(window, min_periods=max(40, window // 2)).cov(pair["m"])
    var = pair["m"].rolling(window, min_periods=max(40, window // 2)).var(ddof=0)
    beta = (cov / var.replace(0.0, np.nan)).shift(1)
    return beta.reindex(asset_r.index)


def _vol_normalized_return(r: pd.Series, cfg: Config) -> pd.Series:
    sigma = r.ewm(span=cfg.vol_span, adjust=False, min_periods=cfg.vol_span).std(bias=False).shift(1)
    z = r / sigma.replace(0.0, np.nan)
    mu = z.rolling(cfg.demean_window, min_periods=max(60, cfg.demean_window // 2)).mean().shift(1)
    z0 = (z - mu).clip(-cfg.z_clip, cfg.z_clip)
    return z0


def _spectral_bank(z: pd.Series, cfg: Config) -> pd.DataFrame:
    """Estimate Sepp-Lucic Psi_nu efficiently from EWMA covariance.

    Paper identities:
      Psi_nu = sum_{m>=1} nu^m rho(m)
      Cov[z_t, L_nu(z_{t-1})] / Var[z_t] = A_nu
      A_nu = ((1-nu)/nu) * Psi_nu

    Hence Psi_hat = nu/(1-nu) * rolling_cov(z_t, L_{t-1}) / rolling_var(z_t).
    """
    minp = max(100, int(cfg.spectral_window * cfg.min_history_frac))
    var_z = z.rolling(cfg.spectral_window, min_periods=minp).var(ddof=0)
    out = pd.DataFrame(index=z.index)
    for span in cfg.spectral_spans:
        if span <= 1:
            raise ValueError("spectral spans must be > 1")
        nu = (span - 1.0) / (span + 1.0)  # eta=1-nu=2/(span+1)
        lam_prev = z.ewm(alpha=1.0 - nu, adjust=False, min_periods=span).mean().shift(1)
        cov = z.rolling(cfg.spectral_window, min_periods=minp).cov(lam_prev)
        a_hat = cov / var_z.replace(0.0, np.nan)
        psi = (nu / (1.0 - nu)) * a_hat
        out[f"psi_{span}"] = psi
        out[f"poisson_excess_{span}"] = 2.0 * psi
    return out


def _future_sum(x: pd.Series, h: int) -> pd.Series:
    # At t: sum x[t+1],...,x[t+h].
    return x.shift(-h).rolling(h, min_periods=h).sum()


def build_panel(
    ohlcv: Dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    if cfg.market_ticker not in ohlcv:
        raise RuntimeError(f"Market ticker {cfg.market_ticker} is required but unavailable")

    market_close = ohlcv[cfg.market_ticker]["Close"].dropna()
    market_r = np.log(market_close).diff()
    class_map = dict(zip(universe["ticker"], universe["asset_class"]))
    frames: List[pd.DataFrame] = []

    modes: List[str] = []
    if cfg.include_raw_mode:
        modes.append("raw")
    if cfg.include_market_residual_mode:
        modes.append("market_residual")

    for k, (ticker, d0) in enumerate(ohlcv.items(), start=1):
        if ticker not in class_map:
            continue
        d = d0.copy().dropna(subset=["Close"])
        if len(d) < cfg.spectral_window + max(cfg.horizons) + 100:
            continue
        raw_r = np.log(d["Close"]).diff()
        mkt = market_r.reindex(d.index)

        for mode in modes:
            if mode == "market_residual" and ticker != cfg.market_ticker:
                beta = _rolling_beta(raw_r, mkt, cfg.beta_window)
                r = raw_r - beta * mkt
            else:
                r = raw_r.copy()

            z = _vol_normalized_return(r, cfg)
            spec = _spectral_bank(z, cfg)

            controls = pd.DataFrame(index=d.index)
            controls["vol20"] = raw_r.rolling(20, min_periods=15).std(ddof=0) * math.sqrt(252.0)
            controls["range20"] = np.log(d["High"] / d["Low"]).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean()
            dollar_vol = (d["Close"] * d["Volume"]).where(d["Volume"] > 0)
            controls["log_dollar_volume20"] = np.log(dollar_vol).rolling(20, min_periods=15).mean()

            for h in cfg.horizons:
                if h not in cfg.spectral_spans:
                    raise ValueError(f"Each horizon must have a matching spectral span; missing {h}")
                past = z.rolling(h, min_periods=h).sum() / math.sqrt(h)
                future = _future_sum(z, h) / math.sqrt(h)
                product = past * future

                x = pd.DataFrame(index=d.index)
                x["ticker"] = ticker
                x["asset_class"] = class_map[ticker]
                x["return_mode"] = mode
                x["horizon"] = h
                x["psi_primary"] = spec[f"psi_{h}"]
                x["poisson_excess_primary"] = spec[f"poisson_excess_{h}"]
                for s in cfg.spectral_spans:
                    x[f"psi_{s}"] = spec[f"psi_{s}"]
                x["past_state"] = past
                x["future_state"] = future
                x["phase_product"] = product
                x["phase_label"] = np.where(product.notna(), (product > 0.0).astype(float), np.nan)
                x = x.join(controls)
                x = x.iloc[:: cfg.anchor_stride]
                required = ["psi_primary", "phase_product", "phase_label"]
                x = x.dropna(subset=required)
                x = x.reset_index(names="date")
                frames.append(x)

        if k % 10 == 0:
            print(f"features {k}/{len(ohlcv)} tickers")

    if not frames:
        raise RuntimeError("No panel rows were produced. Increase history or reduce windows.")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["return_mode", "date", "ticker", "horizon"]).reset_index(drop=True)



def build_panel_native(
    ohlcv: Dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Build the panel using batched C++/OpenMP or CUDA kernels.

    Pandas remains responsible for date alignment and the causal volatility/beta
    preprocessing. The expensive spectral rolling covariance plus all past/future
    horizon sums are fused into mrspd-native and executed in batches.
    """
    if _mrspd_native is None:
        raise RuntimeError(
            "mrspd-native is not installed. Run `python -m pip install ./native` "
            "or use --native-backend off."
        )
    if cfg.market_ticker not in ohlcv:
        raise RuntimeError(f"Market ticker {cfg.market_ticker} is required but unavailable")

    market_close = ohlcv[cfg.market_ticker]["Close"].dropna()
    market_r = np.log(market_close).diff()
    class_map = dict(zip(universe["ticker"], universe["asset_class"]))

    modes: List[str] = []
    if cfg.include_raw_mode:
        modes.append("raw")
    if cfg.include_market_residual_mode:
        modes.append("market_residual")

    # Preprocessing remains causal and identical to the reference implementation.
    entries: List[dict] = []
    for k, (ticker, d0) in enumerate(ohlcv.items(), start=1):
        if ticker not in class_map:
            continue
        d = d0.copy().dropna(subset=["Close"])
        if len(d) < cfg.spectral_window + max(cfg.horizons) + 100:
            continue
        raw_r = np.log(d["Close"]).diff()
        mkt = market_r.reindex(d.index)

        controls = pd.DataFrame(index=d.index)
        controls["vol20"] = raw_r.rolling(20, min_periods=15).std(ddof=0) * math.sqrt(252.0)
        controls["range20"] = np.log(d["High"] / d["Low"]).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean()
        dollar_vol = (d["Close"] * d["Volume"]).where(d["Volume"] > 0)
        controls["log_dollar_volume20"] = np.log(dollar_vol).rolling(20, min_periods=15).mean()

        for mode in modes:
            if mode == "market_residual" and ticker != cfg.market_ticker:
                beta = _rolling_beta(raw_r, mkt, cfg.beta_window)
                r = raw_r - beta * mkt
            else:
                r = raw_r.copy()
            z = _vol_normalized_return(r, cfg)
            entries.append({
                "ticker": ticker,
                "asset_class": class_map[ticker],
                "return_mode": mode,
                "date": d.index.to_numpy(),
                "z": z.to_numpy(dtype=np.float64, copy=False),
                "controls": controls.to_numpy(dtype=np.float64, copy=False),
            })
        if k % 25 == 0:
            print(f"preprocess {k}/{len(ohlcv)} tickers")

    if not entries:
        raise RuntimeError("No panel rows were produced. Increase history or reduce windows.")

    spans = np.ascontiguousarray(cfg.spectral_spans, dtype=np.int32)
    horizons = np.ascontiguousarray(cfg.horizons, dtype=np.int32)
    span_to_idx = {int(v): i for i, v in enumerate(spans)}
    minp = max(100, int(cfg.spectral_window * cfg.min_history_frac))
    frames: List[pd.DataFrame] = []

    for start in range(0, len(entries), cfg.native_batch_size):
        batch = entries[start : start + cfg.native_batch_size]
        max_t = max(len(e["z"]) for e in batch)
        z2 = np.full((len(batch), max_t), np.nan, dtype=np.float64)
        for j, e in enumerate(batch):
            z2[j, : len(e["z"])] = e["z"]

        psi, past_bank, future_bank = _mrspd_native.feature_bank_batch(
            z2,
            spans,
            cfg.spectral_window,
            minp,
            horizons,
            backend=cfg.native_backend,
        )

        # pandas EWM(ignore_na=False) has special semantics for internal gaps.
        # Daily yfinance series are normally contiguous after warm-up, but for
        # scientific equivalence we fall back only for the rare gappy series.
        for j, e in enumerate(batch):
            zz = e["z"]
            finite = np.isfinite(zz)
            where = np.flatnonzero(finite)
            if where.size and not finite[where[0] : where[-1] + 1].all():
                zref = pd.Series(zz, index=pd.to_datetime(e["date"]))
                sref = _spectral_bank(zref, cfg)
                for si, span in enumerate(spans):
                    psi[j, si, : len(zz)] = sref[f"psi_{int(span)}"].to_numpy(float)

        for j, e in enumerate(batch):
            n = len(e["z"])
            dates = e["date"]
            controls = e["controls"]
            # Build the spectral columns once per ticker/mode, then reuse them for every horizon.
            spec_arrays = {f"psi_{int(s)}": psi[j, si, :n] for si, s in enumerate(spans)}
            for hi, h0 in enumerate(horizons):
                h = int(h0)
                si = span_to_idx[h]
                past = past_bank[j, hi, :n] / math.sqrt(h)
                future = future_bank[j, hi, :n] / math.sqrt(h)
                product = past * future
                primary = psi[j, si, :n]

                idx = np.arange(0, n, cfg.anchor_stride, dtype=np.int64)
                keep = np.isfinite(primary[idx]) & np.isfinite(product[idx])
                idx = idx[keep]
                if idx.size == 0:
                    continue

                data = {
                    "date": dates[idx],
                    "ticker": np.repeat(e["ticker"], idx.size),
                    "asset_class": np.repeat(e["asset_class"], idx.size),
                    "return_mode": np.repeat(e["return_mode"], idx.size),
                    "horizon": np.full(idx.size, h, dtype=np.int32),
                    "psi_primary": primary[idx],
                    "poisson_excess_primary": 2.0 * primary[idx],
                }
                for name, arr in spec_arrays.items():
                    data[name] = arr[idx]
                data["past_state"] = past[idx]
                data["future_state"] = future[idx]
                data["phase_product"] = product[idx]
                data["phase_label"] = (product[idx] > 0.0).astype(np.float64)
                data["vol20"] = controls[idx, 0]
                data["range20"] = controls[idx, 1]
                data["log_dollar_volume20"] = controls[idx, 2]
                frames.append(pd.DataFrame(data))

        print(
            f"native features {min(start + len(batch), len(entries))}/{len(entries)} series "
            f"backend={cfg.native_backend}"
        )

    if not frames:
        raise RuntimeError("Native feature engine produced no valid rows")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["return_mode", "date", "ticker", "horizon"]).reset_index(drop=True)

def _weighted_mean_by_date(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("date")["date"].transform("size").astype(float)
    w = 1.0 / counts.to_numpy()
    return w / np.mean(w)


def _safe_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=w))


def _boundary_from_iso(model: IsotonicRegression, lo: float, hi: float) -> float:
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float("nan")
    grid = np.linspace(lo, hi, 4096)
    p = model.predict(grid)
    idx = np.flatnonzero(p >= 0.5)
    if len(idx) == 0 or idx[0] == 0:
        return float("nan") if len(idx) == 0 else float(grid[0])
    j = int(idx[0])
    x0, x1 = grid[j - 1], grid[j]
    p0, p1 = p[j - 1], p[j]
    if p1 == p0:
        return float(x1)
    return float(x0 + (0.5 - p0) * (x1 - x0) / (p1 - p0))


def _fit_isotonic(train: pd.DataFrame) -> Tuple[IsotonicRegression, float, float, float]:
    x = train["psi_primary"].to_numpy(float)
    y = train["phase_label"].to_numpy(int)
    w = _weighted_mean_by_date(train)
    lo, hi = np.nanquantile(x, [0.005, 0.995])
    xfit = np.clip(x, lo, hi)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(xfit, y, sample_weight=w)
    boundary = _boundary_from_iso(model, lo, hi)
    return model, float(lo), float(hi), boundary


def _walkforward_date_blocks(df: pd.DataFrame, cfg: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    dates = np.array(sorted(pd.to_datetime(df["date"].unique())))
    n = len(dates)
    start = max(1, int(cfg.min_train_frac * n))
    if n - start < cfg.n_walkforward_splits * 5:
        raise RuntimeError("Not enough dates for requested walk-forward splits")
    edges = np.linspace(start, n, cfg.n_walkforward_splits + 1, dtype=int)
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(cfg.n_walkforward_splits):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        blocks.append((pd.Timestamp(dates[a]), pd.Timestamp(dates[b - 1])))
    return blocks


def evaluate_isotonic_phase(panel: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds: List[pd.DataFrame] = []
    metric_rows: List[dict] = []
    boundary_rows: List[dict] = []
    loaco_rows: List[dict] = []

    for mode, pm in panel.groupby("return_mode", sort=False):
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, start=1):
            # Universal model: strongest purge because training pools all horizons.
            purge = pd.offsets.BDay(max(cfg.horizons))
            train = pm[pm["date"] < (test_start - purge)]
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)]
            if len(train) < 1000 or len(test) < 200:
                continue

            model, lo, hi, boundary = _fit_isotonic(train)
            p = model.predict(np.clip(test["psi_primary"].to_numpy(float), lo, hi))
            y = test["phase_label"].to_numpy(int)
            wt = _weighted_mean_by_date(test)
            baseline = np.average(train["phase_label"].to_numpy(float), weights=_weighted_mean_by_date(train))
            bp = np.full_like(p, baseline, dtype=float)

            pred = test[["date", "ticker", "asset_class", "return_mode", "horizon", "psi_primary", "phase_label", "phase_product"]].copy()
            pred["fold"] = fold
            pred["model"] = "universal_isotonic"
            pred["p_momentum"] = p
            all_preds.append(pred)

            metric_rows.append({
                "return_mode": mode,
                "fold": fold,
                "horizon": "ALL",
                "model": "universal_isotonic",
                "n": len(test),
                "auc": _safe_auc(y, p, wt),
                "brier": brier_score_loss(y, p, sample_weight=wt),
                "baseline_brier": brier_score_loss(y, bp, sample_weight=wt),
                "brier_skill": 1.0 - brier_score_loss(y, p, sample_weight=wt) / brier_score_loss(y, bp, sample_weight=wt),
                "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=wt),
                "spearman_psi_product": spearmanr(test["psi_primary"], test["phase_product"], nan_policy="omit").statistic,
            })
            boundary_rows.append({
                "return_mode": mode,
                "fold": fold,
                "test_start": test_start,
                "test_end": test_end,
                "horizon": "ALL",
                "model": "universal_isotonic",
                "boundary_psi": boundary,
                "train_n": len(train),
                "test_n": len(test),
            })

            # Horizon-specific models quantify how much is gained by abandoning universality.
            for h in cfg.horizons:
                trh = pm[(pm["horizon"] == h) & (pm["date"] < (test_start - pd.offsets.BDay(h)))]
                teh = pm[(pm["horizon"] == h) & (pm["date"] >= test_start) & (pm["date"] <= test_end)]
                if len(trh) < 400 or len(teh) < 100:
                    continue
                mh, hlo, hhi, hb = _fit_isotonic(trh)
                ph = mh.predict(np.clip(teh["psi_primary"].to_numpy(float), hlo, hhi))
                yh = teh["phase_label"].to_numpy(int)
                wh = _weighted_mean_by_date(teh)
                bh = np.average(trh["phase_label"].to_numpy(float), weights=_weighted_mean_by_date(trh))
                bph = np.full_like(ph, bh, dtype=float)
                metric_rows.append({
                    "return_mode": mode,
                    "fold": fold,
                    "horizon": h,
                    "model": "horizon_isotonic",
                    "n": len(teh),
                    "auc": _safe_auc(yh, ph, wh),
                    "brier": brier_score_loss(yh, ph, sample_weight=wh),
                    "baseline_brier": brier_score_loss(yh, bph, sample_weight=wh),
                    "brier_skill": 1.0 - brier_score_loss(yh, ph, sample_weight=wh) / brier_score_loss(yh, bph, sample_weight=wh),
                    "logloss": log_loss(yh, np.clip(ph, 1e-6, 1 - 1e-6), sample_weight=wh),
                    "spearman_psi_product": spearmanr(teh["psi_primary"], teh["phase_product"], nan_policy="omit").statistic,
                })
                boundary_rows.append({
                    "return_mode": mode,
                    "fold": fold,
                    "test_start": test_start,
                    "test_end": test_end,
                    "horizon": h,
                    "model": "horizon_isotonic",
                    "boundary_psi": hb,
                    "train_n": len(trh),
                    "test_n": len(teh),
                })

            # Time-OOS + asset-class-OOS: fit without the class, test only on the unseen class.
            classes = sorted(test["asset_class"].dropna().unique())
            for cls in classes:
                trc = train[train["asset_class"] != cls]
                tec = test[test["asset_class"] == cls]
                if len(trc) < 1000 or len(tec) < 100:
                    continue
                mc, clo, chi, cb = _fit_isotonic(trc)
                pc = mc.predict(np.clip(tec["psi_primary"].to_numpy(float), clo, chi))
                yc = tec["phase_label"].to_numpy(int)
                wc = _weighted_mean_by_date(tec)
                loaco_rows.append({
                    "return_mode": mode,
                    "fold": fold,
                    "asset_class_held_out": cls,
                    "n": len(tec),
                    "auc": _safe_auc(yc, pc, wc),
                    "brier": brier_score_loss(yc, pc, sample_weight=wc),
                    "boundary_psi_train_without_class": cb,
                })

    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    boundaries = pd.DataFrame(boundary_rows)
    loaco = pd.DataFrame(loaco_rows)
    preds.to_csv(out_dir / "oos_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out_dir / "metrics_isotonic.csv", index=False)
    boundaries.to_csv(out_dir / "boundaries.csv", index=False)
    loaco.to_csv(out_dir / "metrics_leave_one_asset_class_out.csv", index=False)


def evaluate_model_comparison(panel: pd.DataFrame, cfg: Config, out_dir: Path) -> None:
    """Compare spectrum-only information with OHLCV controls in strict time OOS tests."""
    rows: List[dict] = []
    spec_cols = [f"psi_{s}" for s in cfg.spectral_spans]
    control_cols = ["vol20", "range20", "log_dollar_volume20"]

    for mode, pm in panel.groupby("return_mode", sort=False):
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, start=1):
            train = pm[pm["date"] < (test_start - pd.offsets.BDay(max(cfg.horizons)))].copy()
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)].copy()
            if len(train) < 1000 or len(test) < 200:
                continue

            designs = {
                "horizon_only": [],
                "spectral_only": spec_cols,
                "controls_only": control_cols,
                "spectral_plus_controls": spec_cols + control_cols,
            }
            for name, num_cols in designs.items():
                transformers = []
                if num_cols:
                    transformers.append((
                        "num",
                        Pipeline([
                            ("imp", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]),
                        num_cols,
                    ))
                transformers.append(("h", OneHotEncoder(handle_unknown="ignore"), ["horizon"]))
                pre = ColumnTransformer(transformers, remainder="drop")
                clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                wtr = _weighted_mean_by_date(train)
                pipe.fit(train, train["phase_label"].astype(int), clf__sample_weight=wtr)
                p = pipe.predict_proba(test)[:, 1]
                y = test["phase_label"].to_numpy(int)
                wt = _weighted_mean_by_date(test)
                base = np.average(train["phase_label"], weights=wtr)
                base_p = np.full_like(p, base)
                rows.append({
                    "return_mode": mode,
                    "fold": fold,
                    "model": name,
                    "n": len(test),
                    "auc": _safe_auc(y, p, wt),
                    "brier": brier_score_loss(y, p, sample_weight=wt),
                    "baseline_brier": brier_score_loss(y, base_p, sample_weight=wt),
                    "brier_skill": 1.0 - brier_score_loss(y, p, sample_weight=wt) / brier_score_loss(y, base_p, sample_weight=wt),
                    "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), sample_weight=wt),
                })

    pd.DataFrame(rows).to_csv(out_dir / "metrics_model_comparison.csv", index=False)


def make_summary(out_dir: Path) -> dict:
    metrics_path = out_dir / "metrics_isotonic.csv"
    bounds_path = out_dir / "boundaries.csv"
    cmp_path = out_dir / "metrics_model_comparison.csv"
    loaco_path = out_dir / "metrics_leave_one_asset_class_out.csv"
    phase_surface_path = out_dir / "phase_surface_summary.json"

    summary: dict = {}
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        uni = m[m["model"] == "universal_isotonic"]
        summary["universal_isotonic"] = {
            "mean_auc": float(uni["auc"].mean()),
            "mean_brier_skill": float(uni["brier_skill"].mean()),
            "mean_spearman_psi_product": float(uni["spearman_psi_product"].mean()),
            "folds": int(len(uni)),
        }
    if bounds_path.exists():
        b = pd.read_csv(bounds_path)
        bu = b[b["model"] == "universal_isotonic"].copy()
        vals = pd.to_numeric(bu["boundary_psi"], errors="coerce").dropna()
        summary["universal_boundary"] = {
            "mean_psi": float(vals.mean()) if len(vals) else None,
            "std_psi": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "n_finite": int(len(vals)),
        }
        hs = b[b["model"] == "horizon_isotonic"].copy()
        hs["boundary_psi"] = pd.to_numeric(hs["boundary_psi"], errors="coerce")
        if not hs.empty:
            hstats = hs.groupby("horizon")["boundary_psi"].agg(["mean", "std", "count"])
            hstats.to_csv(out_dir / "boundary_by_horizon.csv")
    if cmp_path.exists():
        c = pd.read_csv(cmp_path)
        summary["model_comparison_mean"] = c.groupby(["return_mode", "model"])[["auc", "brier_skill"]].mean().reset_index().to_dict("records")
    if loaco_path.exists():
        l = pd.read_csv(loaco_path)
        if not l.empty:
            summary["leave_one_asset_class_out_mean"] = l.groupby(["return_mode", "asset_class_held_out"])[["auc", "brier"]].mean().reset_index().to_dict("records")
    if phase_surface_path.exists():
        with phase_surface_path.open("r", encoding="utf-8") as f:
            summary["nonmonotone_phase_surface"] = json.load(f)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def make_plots(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pred_path = out_dir / "oos_predictions.csv.gz"
    bound_path = out_dir / "boundaries.csv"
    if pred_path.exists():
        p = pd.read_csv(pred_path, parse_dates=["date"])
        if not p.empty:
            # OOS empirical phase diagram: probability of same-sign continuation by Psi bin and horizon.
            p["psi_bin"] = pd.qcut(p["psi_primary"], q=20, duplicates="drop")
            tab = p.groupby(["horizon", "psi_bin"], observed=True)["phase_label"].mean().unstack()
            fig, ax = plt.subplots(figsize=(11, 6))
            im = ax.imshow(tab.to_numpy(), aspect="auto", origin="lower", vmin=0.35, vmax=0.65)
            ax.set_yticks(range(len(tab.index)))
            ax.set_yticklabels(tab.index.astype(str))
            ax.set_xlabel("Psi quantile bin (low -> high)")
            ax.set_ylabel("Horizon (trading days)")
            ax.set_title("Out-of-sample momentum probability phase diagram")
            fig.colorbar(im, ax=ax, label="P(momentum)")
            fig.tight_layout()
            fig.savefig(out_dir / "phase_diagram_oos.png", dpi=180)
            plt.close(fig)

    if bound_path.exists():
        b = pd.read_csv(bound_path, parse_dates=["test_start", "test_end"])
        bu = b[b["model"] == "universal_isotonic"].copy()
        if not bu.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for mode, g in bu.groupby("return_mode"):
                ax.plot(g["test_start"], pd.to_numeric(g["boundary_psi"], errors="coerce"), marker="o", label=mode)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xlabel("OOS test block start")
            ax.set_ylabel("Estimated universal boundary Psi*")
            ax.set_title("Dynamic spectral phase boundary")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "dynamic_boundary.png", dpi=180)
            plt.close(fig)

        bh = b[b["model"] == "horizon_isotonic"].copy()
        if not bh.empty:
            bh["horizon_num"] = pd.to_numeric(bh["horizon"], errors="coerce")
            bh["boundary_psi"] = pd.to_numeric(bh["boundary_psi"], errors="coerce")
            g = bh.groupby(["return_mode", "horizon_num"])["boundary_psi"].agg(["mean", "std"]).reset_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            for mode, z in g.groupby("return_mode"):
                ax.errorbar(z["horizon_num"], z["mean"], yerr=z["std"], marker="o", capsize=3, label=mode)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_xscale("log")
            ax.set_xlabel("Horizon (trading days, log scale)")
            ax.set_ylabel("Psi* (mean Â± fold std)")
            ax.set_title("Horizon-specific boundaries: universality diagnostic")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "boundary_by_horizon.png", dpi=180)
            plt.close(fig)


def write_manifest(out_dir: Path, cfg: Config, universe: pd.DataFrame, cache_path: Path) -> None:
    payload = {
        "config": asdict(cfg),
        "universe_n": int(len(universe)),
        "universe_sha256": hashlib.sha256(universe.to_csv(index=False).encode()).hexdigest(),
        "cache_path": str(cache_path),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {},
    }
    for name in ["numpy", "pandas", "scipy", "sklearn", "yfinance"]:
        try:
            mod = __import__(name)
            payload["versions"][name] = getattr(mod, "__version__", "unknown")
        except Exception:
            payload["versions"][name] = None
    if _mrspd_native is not None:
        try:
            payload["native"] = _mrspd_native.backend_info()
        except Exception as exc:
            payload["native"] = {"error": str(exc)}
    else:
        payload["native"] = None
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _parse_horizon_csv(text: str) -> tuple[int, ...]:
    vals = tuple(sorted(set(int(x.strip()) for x in str(text).split(",") if x.strip())))
    if not vals or any(x <= 0 for x in vals):
        raise ValueError("--horizons must be a comma-separated list of positive integers")
    return vals


def run_pipeline(args: argparse.Namespace) -> None:
    scan_horizons = _parse_horizon_csv(args.horizons)
    cfg = Config(
        start=args.start,
        end=args.end,
        horizons=scan_horizons,
        spectral_spans=scan_horizons,
        anchor_stride=args.stride,
        market_ticker=args.market_ticker,
        include_raw_mode=not args.residual_only,
        include_market_residual_mode=not args.raw_only,
        native_backend=args.native_backend,
        native_batch_size=args.native_batch_size,
    )
    if args.raw_only and args.residual_only:
        raise ValueError("Choose at most one of --raw-only and --residual-only")

    universe_path = Path(args.universe)
    cache_path = Path(args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = read_universe(universe_path)
    ohlcv = download_ohlcv(universe, cache_path, cfg, force=args.force_download)
    write_manifest(out_dir, cfg, universe, cache_path)

    panel_path = out_dir / "panel.csv.gz"
    if panel_path.exists() and not args.rebuild_panel:
        panel = pd.read_csv(panel_path, parse_dates=["date"])
        panel_horizons = tuple(sorted(int(x) for x in panel["horizon"].dropna().unique()))
        if panel_horizons != cfg.horizons:
            raise RuntimeError(
                f"Existing panel horizons {panel_horizons} do not match requested {cfg.horizons}. "
                "Use --rebuild-panel or a new --out directory."
            )
    else:
        if args.native_backend == "off":
            panel = build_panel(ohlcv, universe, cfg)
        elif _mrspd_native is None:
            if args.native_backend == "auto":
                print("WARNING mrspd-native unavailable; falling back to pandas engine", file=sys.stderr)
                panel = build_panel(ohlcv, universe, cfg)
            else:
                raise RuntimeError("Native backend requested but mrspd-native is not installed")
        else:
            panel = build_panel_native(ohlcv, universe, cfg)
        panel.to_csv(panel_path, index=False, compression="gzip")

    print(f"panel rows: {len(panel):,}")
    print(f"panel horizons: {cfg.horizons}")
    if args.panel_only:
        print(f"PANEL_ONLY: PASS -> {panel_path}")
        return
    evaluate_isotonic_phase(panel, cfg, out_dir)
    evaluate_model_comparison(panel, cfg, out_dir)
    if not args.skip_phase_surface:
        from .phase_surface import run_phase_surface_analysis
        run_phase_surface_analysis(
            panel, cfg, out_dir,
            quantile_bins=args.phase_bins,
            spline_knots=args.phase_knots,
            bootstrap_reps=args.bootstrap_reps,
            max_fit_rows=args.phase_max_fit_rows,
            cert_alpha=args.cert_alpha,
            cert_side_width_q=args.cert_side_width_q,
            cert_side_gap_q=args.cert_side_gap_q,
            cert_min_effect_sigma=args.cert_min_effect_sigma,
            cert_abs_effect=args.cert_abs_effect,
            cert_min_obs_per_side=args.cert_min_obs_per_side,
            cert_min_dates_per_side=args.cert_min_dates_per_side,
            cert_min_folds=args.cert_min_folds,
            cert_min_fold_fraction=args.cert_min_fold_fraction,
            cert_root_cluster_q=args.cert_root_cluster_q,
            cert_max_root_iqr_q=args.cert_max_root_iqr_q,
            cert_single_phase_bins=args.cert_single_phase_bins,
        )
    summary = make_summary(out_dir)
    make_plots(out_dir)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum-Reversal Spectral Phase Diagram")
    p.add_argument("--universe", default="data/universes/discovery_us_mixed.csv")
    p.add_argument("--market-ticker", default="SPY",
                   help="market benchmark ticker used for market_residual; default SPY")
    p.add_argument("--cache", default="data/ohlcv.pkl")
    p.add_argument("--out", default="runs/v001")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default=None, help="yfinance end date is exclusive; omit for latest available")
    p.add_argument("--stride", type=int, default=5, help="sample every N trading rows to reduce dependence and runtime")
    p.add_argument("--horizons", default="5,10,21,42,63,126",
                   help="comma-separated horizons; spectral spans are matched one-for-one")
    p.add_argument("--raw-only", action="store_true")
    p.add_argument("--residual-only", action="store_true")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--rebuild-panel", action="store_true")
    p.add_argument("--panel-only", action="store_true",
                   help="build/cache panel.csv.gz and stop before statistical analyses")
    p.add_argument("--native-backend", choices=["auto", "cpu", "cuda", "off"], default="auto",
                   help="native feature engine: auto selects CUDA for large batches when available")
    p.add_argument("--native-batch-size", type=int, default=128,
                   help="number of ticker/mode series processed in one native batch")
    p.add_argument("--skip-phase-surface", action="store_true",
                   help="skip the non-monotone spline phase-surface extension")
    p.add_argument("--phase-bins", type=int, default=30,
                   help="train-defined Psi quantile bins used for OOS non-parametric curves")
    p.add_argument("--phase-knots", type=int, default=7,
                   help="number of knots in each cubic spline basis")
    p.add_argument("--bootstrap-reps", type=int, default=500,
                   help="moving-block bootstrap replications for paired OOS model deltas")
    p.add_argument("--phase-max-fit-rows", type=int, default=250000,
                   help="date-balanced maximum fit rows per fold for spline models; OOS scoring still uses all test rows")
    p.add_argument("--cert-alpha", type=float, default=0.05,
                   help="significance level for OOS phase-transition certification")
    p.add_argument("--cert-side-width-q", type=float, default=0.08,
                   help="train-Psi quantile width used on each side of a candidate root")
    p.add_argument("--cert-side-gap-q", type=float, default=0.01,
                   help="train-Psi quantile gap kept around the mathematical zero")
    p.add_argument("--cert-min-effect-sigma", type=float, default=0.02,
                   help="minimum |g| on each side as a fraction of train phase_product SD")
    p.add_argument("--cert-abs-effect", type=float, default=0.0,
                   help="absolute floor for the minimum |g| effect size")
    p.add_argument("--cert-min-obs-per-side", type=int, default=250)
    p.add_argument("--cert-min-dates-per-side", type=int, default=40)
    p.add_argument("--cert-min-folds", type=int, default=3)
    p.add_argument("--cert-min-fold-fraction", type=float, default=0.75)
    p.add_argument("--cert-root-cluster-q", type=float, default=0.06,
                   help="maximum quantile distance for matching roots across folds")
    p.add_argument("--cert-max-root-iqr-q", type=float, default=0.05,
                   help="maximum cross-fold IQR of a certified root in Psi-quantile space")
    p.add_argument("--cert-single-phase-bins", type=int, default=5,
                   help="number of OOS Psi bands required to share one sign for R-only/M-only certification")
    return p.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()


from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .critical_horizon import DEFAULT_CRITICAL_HORIZONS, _find_onset, _holm_adjust, _phase_path_local
from .phase_surface import (
    PhaseSurfaceConfig,
    SplineSurface,
    _balanced_fit_sample,
    _date_weights,
    _moving_block_bootstrap,
    _roots_from_curve,
    _walkforward_date_blocks,
    _weighted_std,
)

PROTOCOL_VERSION = "MRSPD-TRANSPORT-TAIL-VALIDATION-v9"
DEFAULT_MEASUREMENT_HORIZONS: tuple[int, ...] = (
    63, 72, 84, 96, 105, 112, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 140, 147, 168,
)
DEFAULT_COARSE_HORIZONS: tuple[int, ...] = tuple(DEFAULT_CRITICAL_HORIZONS)
DEFAULT_TAIL_WIDTH = 0.25
DEFAULT_TAIL_BINS = 2
STREAM_CHUNK_SIZE = 250_000
PANEL_COLUMNS = ["date", "ticker", "return_mode", "horizon", "psi_primary", "phase_product"]


@dataclass(frozen=True)
class TransportConfig:
    spline_knots: int = 7
    bootstrap_reps: int = 1000
    max_fit_rows: int = 250_000
    alpha: float = 0.05
    min_effect_sigma: float = 0.02
    abs_effect: float = 0.0
    min_obs_per_tail_bin: int = 250
    min_dates_per_tail_bin: int = 40
    min_folds: int = 3
    min_fold_fraction: float = 0.75
    tail_width: float = DEFAULT_TAIL_WIDTH
    tail_bins: int = DEFAULT_TAIL_BINS
    hc_min_consecutive: int = 3
    hc_min_tail_fraction: float = 0.80
    root_grid_size: int = 801
    root_qlo: float = 0.01
    root_qhi: float = 0.99
    random_seed: int = 20260812


def canonical_json_sha256(doc: dict) -> str:
    payload = dict(doc)
    payload.pop("protocol_sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_ticker(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _compact_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    z = chunk.copy()
    z["date"] = pd.to_datetime(z["date"], errors="coerce").astype("datetime64[ns]")
    z["ticker"] = z["ticker"].astype(str).map(normalize_ticker)
    z["return_mode"] = z["return_mode"].astype("category")
    z["horizon"] = pd.to_numeric(z["horizon"], errors="coerce").astype("int16")
    z["psi_primary"] = pd.to_numeric(z["psi_primary"], errors="coerce").astype("float64")
    z["phase_product"] = pd.to_numeric(z["phase_product"], errors="coerce").astype("float64")
    return z


def read_panel_streaming(path: Path, *, chunksize: int = STREAM_CHUNK_SIZE, label: str = "panel") -> pd.DataFrame:
    path = Path(path)
    parts: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(path, usecols=PANEL_COLUMNS, chunksize=int(chunksize)):
        z = _compact_chunk(chunk)
        total += len(z)
        parts.append(z)
        if total % 5_000_000 < len(z):
            print(f"transport-load {label}: {total:,} rows")
    if not parts:
        raise ValueError(f"empty panel: {path}")
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    frame["ticker"] = frame["ticker"].astype("category")
    frame["return_mode"] = frame["return_mode"].astype("category")
    print(f"transport-load {label}: rows={len(frame):,} cols={len(frame.columns)} ram={frame.memory_usage(deep=True).sum()/2**20:.1f} MiB")
    return frame


def panel_metadata_streaming(path: Path, *, chunksize: int = STREAM_CHUNK_SIZE) -> dict:
    tickers: set[str] = set()
    modes: set[str] = set()
    horizons: set[int] = set()
    n = 0
    date_min = None
    date_max = None
    for c in pd.read_csv(path, usecols=["date", "ticker", "return_mode", "horizon"], chunksize=int(chunksize)):
        n += len(c)
        tickers.update(normalize_ticker(x) for x in c["ticker"].dropna().unique())
        modes.update(str(x) for x in c["return_mode"].dropna().unique())
        horizons.update(int(x) for x in c["horizon"].dropna().unique())
        d = pd.to_datetime(c["date"], errors="coerce")
        if d.notna().any():
            lo, hi = d.min(), d.max()
            date_min = lo if date_min is None or lo < date_min else date_min
            date_max = hi if date_max is None or hi > date_max else date_max
    ts = sorted(tickers)
    h = hashlib.sha256("\n".join(ts).encode("utf-8")).hexdigest()
    return {
        "rows": int(n),
        "ticker_count": len(ts),
        "tickers": ts,
        "ticker_set_sha256": h,
        "date_min": str(date_min.date()) if date_min is not None else None,
        "date_max": str(date_max.date()) if date_max is not None else None,
        "horizons": sorted(horizons),
        "return_modes": sorted(modes),
    }


def add_cross_sectional_percentile(panel: pd.DataFrame) -> pd.DataFrame:
    """Exact q_Psi = cross-sectional percentile within universe x date x mode x h."""
    z = panel
    q = z.groupby(["date", "return_mode", "horizon"], observed=True, sort=False)["psi_primary"].rank(
        method="average", pct=True
    )
    z["q_psi"] = q.astype("float32")
    return z


def attach_metadata(panel: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    if metadata_path is None:
        return panel
    m = pd.read_csv(metadata_path)
    if "ticker" not in m.columns:
        raise ValueError("metadata CSV must contain ticker")
    m = m.copy()
    m["ticker"] = m["ticker"].map(normalize_ticker)
    keep = [c for c in ["ticker", "asset_class", "source_index", "sector", "cap_bucket"] if c in m.columns]
    m = m[keep].drop_duplicates("ticker")
    if "cap_bucket" not in m.columns and "source_index" in m.columns:
        mapping = {"sp500": "largecap", "sp400": "midcap", "sp600": "smallcap"}
        m["cap_bucket"] = m["source_index"].astype(str).str.lower().map(mapping)
    z = panel
    # Map metadata through ticker category codes. This avoids materializing a
    # 20-30M-row Python-string Series merely to attach sector/cap labels.
    if not isinstance(z["ticker"].dtype, pd.CategoricalDtype):
        z["ticker"] = z["ticker"].astype("category")
    ticker_categories = z["ticker"].cat.categories.astype(str)
    ticker_codes = z["ticker"].cat.codes.to_numpy(copy=False)
    for c in ["asset_class", "source_index", "sector", "cap_bucket"]:
        if c in m.columns:
            mp = dict(zip(m["ticker"].astype(str), m[c]))
            mapped_by_ticker = pd.Categorical(pd.Series(ticker_categories).map(mp))
            code_lut = mapped_by_ticker.codes
            row_codes = np.where(ticker_codes >= 0, code_lut[np.maximum(ticker_codes, 0)], -1)
            z[c] = pd.Categorical.from_codes(row_codes, categories=mapped_by_ticker.categories)
    return z


def psi_quantile_table(panel: pd.DataFrame, universe_label: str) -> pd.DataFrame:
    probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows: list[dict] = []
    for (mode, h), g in panel.groupby(["return_mode", "horizon"], observed=True, sort=True):
        x = g["psi_primary"].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        qs = np.quantile(x, probs)
        row = {
            "universe": universe_label,
            "return_mode": str(mode),
            "horizon": int(h),
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        }
        for p, v in zip(probs, qs):
            row[f"q{int(round(100*p)):02d}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def _cfg_namespace(horizons: Sequence[int], *, seed: int, splits: int, train_frac: float, stride: int):
    return SimpleNamespace(
        horizons=tuple(horizons),
        spectral_spans=tuple(horizons),
        n_walkforward_splits=int(splits),
        min_train_frac=float(train_frac),
        anchor_stride=int(stride),
        random_seed=int(seed),
    )


def normalized_root_and_curve_analysis(
    panel: pd.DataFrame,
    *,
    universe_label: str,
    horizons: Sequence[int],
    cfg,
    tcfg: TransportConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    curve_rows: list[dict] = []
    qbin_rows: list[dict] = []
    hs = tuple(sorted(set(int(x) for x in horizons)))
    q_grid = np.linspace(tcfg.root_qlo, tcfg.root_qhi, tcfg.root_grid_size)
    max_h = max(hs)
    for mi, (mode, pm0) in enumerate(panel.groupby("return_mode", observed=True, sort=False)):
        pm = pm0.loc[pm0["horizon"].isin(hs), ["date", "horizon", "q_psi", "phase_product"]].copy()
        pm["psi_primary"] = pm["q_psi"].astype("float64")
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, 1):
            purge = pd.offsets.BDay(max_h)
            train = pm[pm["date"] < test_start - purge]
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)]
            if len(train) < 1000 or len(test) < 200:
                continue
            seed = tcfg.random_seed + 100_000*(mi+1) + fold
            fit = _balanced_fit_sample(train, tcfg.max_fit_rows, seed)
            surf = SplineSurface(
                task="continuous", interaction=True, context=False,
                n_knots=tcfg.spline_knots, degree=3, ridge_alpha=4.0, logistic_c=0.5,
            ).fit(fit, _date_weights(fit))
            for h in hs:
                ev = pd.DataFrame({"psi_primary": q_grid, "horizon": np.full(len(q_grid), h, float)})
                gh = surf.predict(ev)
                roots = _roots_from_curve(q_grid, gh)
                rows.append({
                    "universe": universe_label, "return_mode": str(mode), "fold": fold,
                    "test_start": test_start, "test_end": test_end, "horizon": int(h),
                    "n_roots": len(roots), "phase_path": _phase_path_local(q_grid, gh, roots),
                    "roots_q_json": json.dumps([float(r["psi_star"]) for r in roots]),
                })
                for qv, gv in zip(q_grid, gh):
                    curve_rows.append({
                        "universe": universe_label, "return_mode": str(mode), "fold": fold,
                        "horizon": int(h), "q_psi": float(qv), "g_hat": float(gv),
                    })
                th = test[test["horizon"] == h]
                if not th.empty:
                    b = pd.cut(th["q_psi"].astype(float), bins=np.linspace(0,1,21), include_lowest=True, labels=False)
                    tmp = th.assign(qbin=b).dropna(subset=["qbin"])
                    agg = tmp.groupby("qbin", observed=True)["phase_product"].agg(["mean","count"]).reset_index()
                    for _, rr in agg.iterrows():
                        qb = int(rr["qbin"])
                        qbin_rows.append({
                            "universe": universe_label, "return_mode": str(mode), "fold": fold,
                            "horizon": int(h), "qbin": qb, "q_mid": (qb+0.5)/20.0,
                            "phase_mean": float(rr["mean"]), "n": int(rr["count"]),
                        })
            print(f"transport-roots {universe_label} {mode} fold {fold}/{len(blocks)} complete")
    roots = pd.DataFrame(rows)
    curves = pd.DataFrame(curve_rows)
    qbins = pd.DataFrame(qbin_rows)
    if not curves.empty:
        curves = curves.groupby(["universe","return_mode","horizon","q_psi"], as_index=False).agg(
            g_hat_mean=("g_hat","mean"), g_hat_sd=("g_hat","std"), folds=("fold","nunique")
        )
    return roots, curves, qbins


def _tail_edges(direction: str, width: float, bins: int) -> np.ndarray:
    direction = str(direction).lower()
    if direction == "lower":
        return np.linspace(0.0, float(width), int(bins)+1)
    if direction == "upper":
        return np.linspace(1.0-float(width), 1.0, int(bins)+1)
    raise ValueError("tail direction must be lower or upper")


def _tail_fold_certification(
    train_h: pd.DataFrame,
    test_h: pd.DataFrame,
    *,
    direction: str,
    tcfg: TransportConfig,
    block_len: int,
    seed: int,
) -> dict:
    scale = _weighted_std(train_h["phase_product"].to_numpy(float), _date_weights(train_h))
    eps = max(tcfg.abs_effect, tcfg.min_effect_sigma * scale if np.isfinite(scale) else 0.0)
    edges = _tail_edges(direction, tcfg.tail_width, tcfg.tail_bins)
    details: list[dict] = []
    pvals: list[float] = []
    all_negative = True
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if b == len(edges)-2:
            mask = (test_h["q_psi"] >= lo) & (test_h["q_psi"] <= hi)
        else:
            mask = (test_h["q_psi"] >= lo) & (test_h["q_psi"] < hi)
        gb = test_h.loc[mask]
        if len(gb) < tcfg.min_obs_per_tail_bin or gb["date"].nunique() < tcfg.min_dates_per_tail_bin:
            return {
                "eligible": False, "reason": "insufficient_tail_bin_support", "tail_negative": False,
                "epsilon": float(eps), "candidate_p_iut": np.nan, "tail_bins_json": json.dumps(details),
            }
        date_means = gb.groupby("date", sort=True)["phase_product"].mean().to_numpy(float)
        st = _moving_block_bootstrap(date_means, block_len, tcfg.bootstrap_reps, np.random.default_rng(seed+101*(b+1)))
        if st["estimate"] is None or st["ci_hi"] is None or st["p_lt0"] is None:
            return {
                "eligible": False, "reason": "bootstrap_unavailable", "tail_negative": False,
                "epsilon": float(eps), "candidate_p_iut": np.nan, "tail_bins_json": json.dumps(details),
            }
        passed = bool(float(st["estimate"]) <= -float(eps) and float(st["ci_hi"]) < 0.0)
        all_negative &= passed
        pvals.append(float(st["p_lt0"]))
        details.append({
            "bin": b, "q_lo": float(lo), "q_hi": float(hi),
            "estimate": float(st["estimate"]), "ci_lo": float(st["ci_lo"]), "ci_hi": float(st["ci_hi"]),
            "p_lt0": float(st["p_lt0"]), "n": int(len(gb)), "n_dates": int(gb["date"].nunique()),
            "effect_floor_pass": bool(float(st["estimate"]) <= -float(eps)),
            "ci_negative_pass": bool(float(st["ci_hi"]) < 0.0),
        })
    p_iut = max(pvals) if len(pvals) == tcfg.tail_bins else np.nan
    return {
        "eligible": True, "reason": "", "tail_negative": bool(all_negative and np.isfinite(p_iut) and p_iut <= tcfg.alpha),
        "epsilon": float(eps), "candidate_p_iut": float(p_iut), "tail_bins_json": json.dumps(details),
    }


def _apply_holm_tail(folds: pd.DataFrame, alpha: float) -> pd.DataFrame:
    z = folds.copy()
    z["p_holm_horizon_family"] = np.nan
    z["passes_horizon_holm"] = False
    z["certified_tail_negative_fold"] = False
    for _, idx in z.groupby(["universe","return_mode","fold","tail_direction"], sort=False).groups.items():
        loc = list(idx)
        adj = _holm_adjust(z.loc[loc,"candidate_p_iut"].to_numpy(float))
        z.loc[loc,"p_holm_horizon_family"] = adj
        z.loc[loc,"passes_horizon_holm"] = np.isfinite(adj) & (adj <= alpha)
    z["certified_tail_negative_fold"] = (
        z["eligible"].fillna(False).astype(bool)
        & z["tail_negative"].fillna(False).astype(bool)
        & z["passes_horizon_holm"].fillna(False).astype(bool)
    )
    return z


def evaluate_tail_direction(
    panel: pd.DataFrame,
    *,
    universe_label: str,
    direction: str,
    horizons: Sequence[int],
    cfg,
    tcfg: TransportConfig,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    hs = tuple(sorted(set(int(x) for x in horizons)))
    max_h = max(hs)
    block_len = max(1, int(math.ceil(max_h / int(cfg.anchor_stride))))
    rows: list[dict] = []
    for mi, (mode, pm) in enumerate(panel.groupby("return_mode", observed=True, sort=False)):
        pm = pm[pm["horizon"].isin(hs)]
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold,(test_start,test_end) in enumerate(blocks,1):
            purge = pd.offsets.BDay(max_h)
            train = pm[pm["date"] < test_start-purge]
            test = pm[(pm["date"]>=test_start)&(pm["date"]<=test_end)]
            for h in hs:
                tr = train[train["horizon"]==h]
                te = test[test["horizon"]==h]
                if len(tr)<200 or len(te)<100:
                    cert={"eligible":False,"reason":"insufficient_horizon_rows","tail_negative":False,"candidate_p_iut":np.nan,"epsilon":np.nan,"tail_bins_json":"[]"}
                else:
                    cert=_tail_fold_certification(
                        tr,te,direction=direction,tcfg=tcfg,block_len=block_len,
                        seed=tcfg.random_seed + 200_000*(mi+1)+10_000*fold+int(h),
                    )
                rows.append({
                    "universe":universe_label,"return_mode":str(mode),"fold":fold,
                    "test_start":test_start,"test_end":test_end,"horizon":int(h),
                    "tail_direction":direction,"tail_width":tcfg.tail_width,**cert,
                })
    folds=_apply_holm_tail(pd.DataFrame(rows),tcfg.alpha)
    regs=[]
    for (u,mode,d,h),g in folds.groupby(["universe","return_mode","tail_direction","horizon"],sort=True):
        total=int(g["fold"].nunique()); req=max(tcfg.min_folds,int(math.ceil(tcfg.min_fold_fraction*total)))
        sup=int(g["certified_tail_negative_fold"].sum())
        regs.append({"universe":u,"return_mode":mode,"tail_direction":d,"horizon":int(h),
                     "folds_total":total,"min_folds_required":req,"negative_folds_support":sup,
                     "support_fraction":sup/total if total else np.nan,"certified_tail_negative":bool(sup>=req)})
    regimes=pd.DataFrame(regs)
    hc_rows=[]
    for (u,mode,d),g in regimes.groupby(["universe","return_mode","tail_direction"],sort=False):
        g=g.sort_values("horizon")
        est=_find_onset(g["horizon"].to_numpy(int),g["certified_tail_negative"].to_numpy(bool),
                        np.zeros(len(g),dtype=bool),min_consecutive=tcfg.hc_min_consecutive,
                        min_tail_fraction=tcfg.hc_min_tail_fraction)
        hc_rows.append({"universe":u,"return_mode":mode,"tail_direction":d,**est})
    hc=pd.DataFrame(hc_rows)
    consensus={"identified":False}
    if not hc.empty and hc["identified"].astype(bool).all():
        upper=int(hc["hc_grid"].max())
        lower_vals=hc["hc_lower_exclusive"].dropna()
        lower=int(lower_vals.max()) if len(lower_vals) else None
        consensus={"identified":True,"hc_grid":upper,"hc_lower_exclusive":lower,"hc_upper_inclusive":upper,
                   "hc_midpoint":0.5*(lower+upper) if lower is not None else float(upper),
                   "modes":hc["return_mode"].tolist()}
    return folds,regimes,hc,consensus


def select_transport_hypothesis(candidate_summaries: list[dict], coarse_horizons: Sequence[int]) -> dict:
    """Choose lower/upper quartile by predeclared transportability ordering."""
    eligible=[]
    for c in candidate_summaries:
        dc=c["discovery_consensus"]; vc=c["validation_consensus"]
        if dc.get("identified") and vc.get("identified"):
            delta=abs(int(dc["hc_grid"])-int(vc["hc_grid"]))
            eligible.append((delta, str(c["tail_direction"]), c))
    if not eligible:
        return {"selected":False,"reason":"no_quartile_tail_has_persistent_negative_onset_in_both_universes_and_both_modes"}
    eligible.sort(key=lambda x:(x[0],x[1]))
    delta,_,best=eligible[0]
    d=int(best["discovery_consensus"]["hc_grid"]); v=int(best["validation_consensus"]["hc_grid"])
    hs=sorted(set(int(x) for x in coarse_horizons))
    envelope=[min(d,v),max(d,v)]
    # One *local* original coarse-grid step around the transported envelope;
    # do not let a distant 147->168 gap silently widen a ~126-day endpoint.
    local_gaps=[]
    for target in set(envelope):
        if target in hs:
            i=hs.index(target)
            if i>0: local_gaps.append(hs[i]-hs[i-1])
            if i+1<len(hs): local_gaps.append(hs[i+1]-hs[i])
    tol=max(local_gaps) if local_gaps else 7
    return {
        "selected":True,
        "tail_direction":best["tail_direction"],
        "tail_width":best["tail_width"],
        "tail_bins":best["tail_bins"],
        "discovery_hc_grid":d,
        "second_universe_hc_grid":v,
        "transport_delta_days":delta,
        "expected_hc_envelope":[int(envelope[0]),int(envelope[1])],
        "localization_tolerance_days":int(tol),
        "selection_rule":"among fixed lower/upper quartile tails, require both universes x both modes identified; choose minimum absolute consensus-hc gap, lexical direction tie-break",
    }


def subgroup_tail_hc(
    panel: pd.DataFrame,
    *,
    universe_label: str,
    direction: str,
    horizons: Sequence[int],
    cfg,
    tcfg: TransportConfig,
    subgroup_columns: Sequence[str] = ("source_index","cap_bucket","sector","asset_class"),
    min_tickers: int = 25,
) -> pd.DataFrame:
    rows=[]
    for col in subgroup_columns:
        if col not in panel.columns:
            continue
        for value,g in panel.groupby(col, observed=True, sort=True):
            if pd.isna(value):
                continue
            nt=int(g["ticker"].nunique()) if "ticker" in g.columns else 0
            if nt<min_tickers:
                continue
            # Exploratory subgroup analysis gets looser row-count floor only to avoid structural NA;
            # alpha/effect/bootstrap stay unchanged and results are explicitly non-confirmatory.
            sgcfg=TransportConfig(**{**tcfg.__dict__,"min_obs_per_tail_bin":min(tcfg.min_obs_per_tail_bin,100)})
            _,_,hc,cons=evaluate_tail_direction(g,universe_label=f"{universe_label}:{col}={value}",direction=direction,horizons=horizons,cfg=cfg,tcfg=sgcfg)
            rows.append({"universe":universe_label,"subgroup_type":col,"subgroup":str(value),"ticker_count":nt,
                         "tail_direction":direction,"consensus_identified":bool(cons.get("identified")),
                         "consensus_hc_grid":cons.get("hc_grid"),"per_mode_json":hc.to_json(orient="records")})
    return pd.DataFrame(rows)


def run_transport_exploratory(
    *,
    discovery_panel_path: Path,
    validation_panel_path: Path,
    out_dir: Path,
    discovery_metadata_path: Path | None = None,
    validation_metadata_path: Path | None = None,
    horizons: Sequence[int] = DEFAULT_COARSE_HORIZONS,
    bootstrap_reps: int = 1000,
    max_fit_rows: int = 250_000,
    alpha: float = 0.05,
    min_effect_sigma: float = 0.02,
    min_folds: int = 3,
    min_fold_fraction: float = 0.75,
    splits: int = 4,
    min_train_frac: float = 0.50,
    stride: int = 5,
    seed: int = 20260812,
) -> dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    hs=tuple(sorted(set(int(x) for x in horizons)))
    tcfg=TransportConfig(bootstrap_reps=bootstrap_reps,max_fit_rows=max_fit_rows,alpha=alpha,
                         min_effect_sigma=min_effect_sigma,min_folds=min_folds,min_fold_fraction=min_fold_fraction,
                         random_seed=seed)
    cfg=_cfg_namespace(hs,seed=seed,splits=splits,train_frac=min_train_frac,stride=stride)

    all_quant=[]; all_roots=[]; all_curves=[]; all_qbins=[]; all_folds=[]; all_regs=[]; all_hc=[]
    per_universe_cons: dict[str,dict[str,dict]]={}
    second_panel_for_subgroup = None
    for label,path,mpath in [
        ("discovery",discovery_panel_path,discovery_metadata_path),
        ("second_universe",validation_panel_path,validation_metadata_path),
    ]:
        p=read_panel_streaming(Path(path),label=label)
        p=p[p["horizon"].isin(hs)].reset_index(drop=True)
        p=attach_metadata(p,Path(mpath) if mpath else None)
        add_cross_sectional_percentile(p)
        if label == "second_universe":
            second_panel_for_subgroup = p
        all_quant.append(psi_quantile_table(p,label))
        roots,curves,qbins=normalized_root_and_curve_analysis(p,universe_label=label,horizons=hs,cfg=cfg,tcfg=tcfg)
        all_roots.append(roots);all_curves.append(curves);all_qbins.append(qbins)
        per_universe_cons[label]={}
        for direction in ("lower","upper"):
            folds,regs,hc,cons=evaluate_tail_direction(p,universe_label=label,direction=direction,horizons=hs,cfg=cfg,tcfg=tcfg)
            all_folds.append(folds);all_regs.append(regs);all_hc.append(hc)
            per_universe_cons[label][direction]=cons

    quant=pd.concat(all_quant,ignore_index=True); quant.to_csv(out/"psi_quantiles_by_universe_horizon.csv",index=False)
    roots=pd.concat(all_roots,ignore_index=True); roots.to_csv(out/"normalized_root_topology_by_fold.csv",index=False)
    if not roots.empty:
        rd=(roots.groupby(["universe","return_mode","horizon","n_roots","phase_path"],as_index=False)
            .agg(folds=("fold","nunique")))
        rd.to_csv(out/"normalized_root_topology_distribution.csv",index=False)
    curves=pd.concat(all_curves,ignore_index=True); curves.to_csv(out/"normalized_g_curves.csv",index=False)
    qbins=pd.concat(all_qbins,ignore_index=True); qbins.to_csv(out/"normalized_oos_qbin_curves.csv",index=False)
    # Direct curve transport distance on matched q-grid.
    if not curves.empty:
        a=curves[curves["universe"]=="discovery"].rename(columns={"g_hat_mean":"g_discovery"})
        b=curves[curves["universe"]=="second_universe"].rename(columns={"g_hat_mean":"g_second"})
        m=a.merge(b,on=["return_mode","horizon","q_psi"],how="inner")
        if not m.empty:
            dist=(m.assign(sq=(m.g_discovery-m.g_second)**2,ab=lambda x:abs(x.g_discovery-x.g_second))
                  .groupby(["return_mode","horizon"],as_index=False).agg(rmse=("sq",lambda x:float(np.sqrt(np.mean(x)))),mae=("ab","mean")))
            dist.to_csv(out/"normalized_g_curve_transport_distance.csv",index=False)

    folds=pd.concat(all_folds,ignore_index=True); folds.to_csv(out/"tail_candidate_fold_certification.csv",index=False)
    regs=pd.concat(all_regs,ignore_index=True); regs.to_csv(out/"tail_candidate_regimes.csv",index=False)
    hc=pd.concat(all_hc,ignore_index=True); hc.to_csv(out/"tail_candidate_hc.csv",index=False)

    candidates=[]
    for d in ("lower","upper"):
        candidates.append({
            "tail_direction":d,"tail_width":tcfg.tail_width,"tail_bins":tcfg.tail_bins,
            "discovery_consensus":per_universe_cons["discovery"][d],
            "validation_consensus":per_universe_cons["second_universe"][d],
        })
    selected=select_transport_hypothesis(candidates,hs)
    (out/"selected_transport_hypothesis.json").write_text(json.dumps(selected,indent=2,default=str),encoding="utf-8")

    subgroup=pd.DataFrame()
    if selected.get("selected") and second_panel_for_subgroup is not None:
        subgroup=subgroup_tail_hc(second_panel_for_subgroup,universe_label="second_universe",
                                  direction=selected["tail_direction"],horizons=hs,cfg=cfg,tcfg=tcfg)
        subgroup.to_csv(out/"second_universe_subgroup_tail_hc.csv",index=False)

    summary={
        "status":"EXPLORATORY_TRANSPORT_ANALYSIS_ONLY",
        "normalized_coordinate":"q_psi = empirical cross-sectional percentile of psi_primary within each universe x date x return_mode x horizon",
        "coarse_horizons":list(hs),
        "tail_candidates":"fixed lower quartile q<=0.25 and upper quartile q>=0.75",
        "tail_stability_rule":"two percentile sub-bands inside the selected quartile must each have OOS date-mean phase_product <= -epsilon and block-bootstrap upper 95% CI < 0; horizon-family Holm; >=3/4 folds",
        "root_free_required":False,
        "candidate_summaries":candidates,
        "selected_transport_hypothesis":selected,
        "rules":{
            "spline_knots":tcfg.spline_knots,"bootstrap_reps":tcfg.bootstrap_reps,"max_fit_rows":tcfg.max_fit_rows,
            "alpha":tcfg.alpha,"min_effect_sigma":tcfg.min_effect_sigma,"tail_width":tcfg.tail_width,"tail_bins":tcfg.tail_bins,
            "min_folds":tcfg.min_folds,"min_fold_fraction":tcfg.min_fold_fraction,
            "hc_min_consecutive":tcfg.hc_min_consecutive,"hc_min_tail_fraction":tcfg.hc_min_tail_fraction,
            "walkforward_splits":splits,"min_train_frac":min_train_frac,"stride":stride,"seed":seed,
        },
        "guardrail":"All results in this directory are exploratory because the second universe has already been observed. Only a separately frozen v9 protocol may be used on a third untouched universe.",
    }
    (out/"transport_exploratory_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    return summary


def freeze_transport_protocol(
    *,
    exploratory_summary_path: Path,
    selected_hypothesis_path: Path,
    discovery_panel_path: Path,
    second_panel_path: Path,
    out_path: Path,
) -> dict:
    summary=json.loads(Path(exploratory_summary_path).read_text(encoding="utf-8"))
    selected=json.loads(Path(selected_hypothesis_path).read_text(encoding="utf-8"))
    if not selected.get("selected"):
        raise ValueError("No transportable quartile-tail hypothesis was selected; do not freeze a third-universe confirmation protocol")
    dmeta=panel_metadata_streaming(Path(discovery_panel_path)); smeta=panel_metadata_streaming(Path(second_panel_path))
    overlap=sorted(set(dmeta["tickers"]).intersection(smeta["tickers"]))
    if overlap:
        raise ValueError(f"discovery and second universe are not ticker-disjoint: {overlap[:10]}")
    rules=summary["rules"]
    protocol={
        "protocol_version":PROTOCOL_VERSION,
        "status":"FROZEN_BEFORE_THIRD_UNIVERSE_VALIDATION",
        "hypothesis":{
            "statement":"After intra-universe/intra-date percentile normalization of Psi, a preselected spectral quartile tail has a persistent negative phase-product regime across horizons, without requiring the entire spectral domain to be root-free.",
            "q_psi_definition":"empirical cross-sectional percentile of psi_primary within third universe x date x return_mode x horizon",
            "tail_direction":selected["tail_direction"],"tail_width":selected["tail_width"],"tail_bins":selected["tail_bins"],
            "expected_hc_envelope":selected["expected_hc_envelope"],
            "localization_tolerance_days":selected["localization_tolerance_days"],
        },
        "provenance":{
            "discovery_panel_metadata":dmeta,"second_universe_panel_metadata":smeta,
            "exploratory_summary_sha256":hashlib.sha256(Path(exploratory_summary_path).read_bytes()).hexdigest(),
            "selected_hypothesis_sha256":hashlib.sha256(Path(selected_hypothesis_path).read_bytes()).hexdigest(),
        },
        "coarse_horizons":summary["coarse_horizons"],
        "rules":rules,
        "primary_endpoints":{
            "independence":"zero ticker overlap with the union of discovery and second-universe panels",
            "phenomenon":"persistent OOS-certified negative selected quartile tail is identified in both raw and market_residual",
            "localization":"third-universe consensus tail h_c lies within frozen expected envelope expanded by frozen tolerance",
            "overall_pass_requires":["independence","phenomenon","localization"],
        },
        "guardrails":[
            "No third-universe thresholds, tail direction, tail width, horizons, folds, bootstrap reps, or localization tolerance may be supplied by the validation CLI.",
            "The entire spectral domain is NOT required to be root-free; only the frozen percentile tail is tested.",
            "The third universe must have zero ticker overlap with both previously observed universes.",
            "Failure on the third universe remains a valid negative confirmatory result and must not trigger post-hoc threshold changes.",
        ],
    }
    protocol["protocol_sha256"]=canonical_json_sha256(protocol)
    Path(out_path).write_text(json.dumps(protocol,indent=2,default=str),encoding="utf-8")
    return protocol


def evaluate_third_universe(
    *,
    third_panel: pd.DataFrame,
    protocol: dict,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    if protocol.get("protocol_version")!=PROTOCOL_VERSION:
        raise ValueError("unsupported v9 transport protocol")
    if protocol.get("protocol_sha256")!=canonical_json_sha256(protocol):
        raise ValueError("protocol SHA256 mismatch")
    rules=protocol["rules"]
    hs=tuple(int(x) for x in protocol["coarse_horizons"])
    tcfg=TransportConfig(
        spline_knots=int(rules["spline_knots"]),bootstrap_reps=int(rules["bootstrap_reps"]),
        max_fit_rows=int(rules["max_fit_rows"]),alpha=float(rules["alpha"]),min_effect_sigma=float(rules["min_effect_sigma"]),
        tail_width=float(protocol["hypothesis"]["tail_width"]),tail_bins=int(protocol["hypothesis"]["tail_bins"]),
        min_folds=int(rules["min_folds"]),min_fold_fraction=float(rules["min_fold_fraction"]),
        hc_min_consecutive=int(rules["hc_min_consecutive"]),hc_min_tail_fraction=float(rules["hc_min_tail_fraction"]),
        random_seed=int(rules["seed"]),
    )
    cfg=_cfg_namespace(hs,seed=int(rules["seed"]),splits=int(rules["walkforward_splits"]),
                       train_frac=float(rules["min_train_frac"]),stride=int(rules["stride"]))
    return evaluate_tail_direction(third_panel,universe_label="third_universe",direction=protocol["hypothesis"]["tail_direction"],horizons=hs,cfg=cfg,tcfg=tcfg)


def bind_third_universe_protocol(
    *,
    protocol_path: Path,
    third_universe_path: Path,
    out_path: Path,
    min_panel_coverage_fraction: float = 0.80,
) -> dict:
    p=json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if p.get("protocol_version")!=PROTOCOL_VERSION or p.get("protocol_sha256")!=canonical_json_sha256(p):
        raise ValueError("invalid base v9 protocol")
    u=pd.read_csv(third_universe_path)
    if not {"ticker","asset_class"}.issubset(u.columns):
        raise ValueError("third universe must contain ticker,asset_class")
    tickers=sorted(set(normalize_ticker(x) for x in u["ticker"].dropna()))
    if len(tickers)<50:
        raise ValueError("third universe is too small to bind")
    q=dict(p); q.pop("protocol_sha256",None)
    q["status"]="FROZEN_WITH_THIRD_UNIVERSE_BEFORE_PANEL_ANALYSIS"
    q["third_universe_design"]={
        "ticker_count":len(tickers),"tickers":tickers,
        "ticker_set_sha256":hashlib.sha256("\n".join(tickers).encode("utf-8")).hexdigest(),
        "min_panel_coverage_fraction":float(min_panel_coverage_fraction),
        "universe_file_sha256":hashlib.sha256(Path(third_universe_path).read_bytes()).hexdigest(),
    }
    q["guardrails"]=list(q.get("guardrails",[]))+[
        "The third-universe ticker list is hash-bound before OHLCV panel construction; the realized panel may only lose tickers because of data availability, never add or replace them.",
    ]
    q["protocol_sha256"]=canonical_json_sha256(q)
    Path(out_path).write_text(json.dumps(q,indent=2,default=str),encoding="utf-8")
    return q


def audit_bound_third_panel(protocol: dict, third_metadata: dict) -> dict:
    design=protocol.get("third_universe_design")
    if not design:
        raise ValueError("third-universe validation requires a protocol bound to a third-universe ticker list")
    allowed=set(design["tickers"]); got=set(third_metadata["tickers"])
    extras=sorted(got-allowed); coverage=len(got)/float(max(1,len(allowed)))
    passed=(not extras) and coverage+1e-15>=float(design["min_panel_coverage_fraction"])
    return {"pass":bool(passed),"bound_tickers":len(allowed),"panel_tickers":len(got),"coverage_fraction":coverage,
            "min_coverage_fraction":float(design["min_panel_coverage_fraction"]),"unexpected_tickers":extras[:20]}

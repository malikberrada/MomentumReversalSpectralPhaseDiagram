from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "MRSPD-ROBUST-PERCENTILE-TRANSPORT-v9.2"
DEFAULT_Q0_GRID = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_HORIZONS = (63,72,84,96,105,112,119,126,133,140,147,168,189,210,231,252)
HUBER_C = 1.345
MAD_NORMALIZER = 1.4826

PANEL_COLS = ["date","ticker","return_mode","horizon","psi_primary","phase_product"]


def canonical_json_sha256(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bh_holm(pvals: Sequence[float], alpha: float) -> list[bool]:
    # Holm step-down FWER control. NaN -> fail.
    n = len(pvals)
    order = sorted(range(n), key=lambda i: (1.0 if not np.isfinite(pvals[i]) else pvals[i]))
    out = [False] * n
    active = True
    for rank, i in enumerate(order):
        p = pvals[i]
        thr = alpha / (n - rank)
        if active and np.isfinite(p) and p <= thr:
            out[i] = True
        else:
            active = False
            out[i] = False
    return out


def load_panel_streaming(path: Path, horizons: Sequence[int] | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    wanted = set(map(int, horizons)) if horizons is not None else None
    rows = 0
    for chunk in pd.read_csv(path, usecols=PANEL_COLS, chunksize=chunksize):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="raise").astype("datetime64[ns]")
        chunk["horizon"] = pd.to_numeric(chunk["horizon"], errors="raise").astype("int16")
        chunk["psi_primary"] = pd.to_numeric(chunk["psi_primary"], errors="coerce").astype("float64")
        chunk["phase_product"] = pd.to_numeric(chunk["phase_product"], errors="coerce").astype("float64")
        chunk["return_mode"] = chunk["return_mode"].astype("category")
        chunk["ticker"] = chunk["ticker"].astype("string")
        if wanted is not None:
            chunk = chunk[chunk["horizon"].isin(wanted)]
        chunk = chunk.dropna(subset=["date","ticker","return_mode","horizon","psi_primary","phase_product"])
        if len(chunk):
            parts.append(chunk)
            rows += len(chunk)
        if rows and rows % 5_000_000 < chunksize:
            print(f"robust-transport-load {path.name}: {rows:,} retained rows", flush=True)
    if not parts:
        raise ValueError(f"No usable rows in {path}")
    out = pd.concat(parts, ignore_index=True)
    out["date"] = out["date"].astype("datetime64[ns]")
    return out


def add_global_percentile_and_robust_response(frame: pd.DataFrame, huber_c: float = HUBER_C) -> pd.DataFrame:
    """Return minimal frame with q_psi and robust phase response.

    Robust response is deterministic cross-sectional Huber clipping at each
    universe x date x return_mode x horizon cell:
      med + clip(y-med, -c*1.4826*MAD, +c*1.4826*MAD)
    No rows, dates, or tickers are removed because of the response magnitude.
    """
    gcols = ["date","return_mode","horizon"]
    d = frame[["date","ticker","return_mode","horizon","psi_primary","phase_product"]].copy()
    d["q_psi"] = d.groupby(gcols, observed=True)["psi_primary"].rank(method="average", pct=True)

    med = d.groupby(gcols, observed=True)["phase_product"].transform("median")
    abs_dev = (d["phase_product"] - med).abs()
    mad = abs_dev.groupby([d[c] for c in gcols], observed=True).transform("median")
    robust_scale = MAD_NORMALIZER * mad

    # Fallback for cells with zero MAD. Keep it deterministic and cell-local.
    std = d.groupby(gcols, observed=True)["phase_product"].transform("std").fillna(0.0)
    robust_scale = robust_scale.where(robust_scale > 0, std)
    robust_scale = robust_scale.where(robust_scale > 0, 1.0)

    delta = d["phase_product"] - med
    cap = huber_c * robust_scale
    d["phase_product_robust"] = med + delta.clip(lower=-cap, upper=cap)
    d["robust_scale_xs"] = robust_scale
    d["was_clipped"] = (delta.abs() > cap)
    return d


def expanding_walkforward_dates(dates: Sequence[pd.Timestamp], splits: int, min_train_frac: float) -> list[tuple[np.ndarray,np.ndarray]]:
    arr = np.array(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())), dtype="datetime64[ns]")
    n = len(arr)
    if n < 10:
        return []
    start = max(1, int(math.floor(min_train_frac * n)))
    test_idx = np.arange(start, n)
    chunks = [c for c in np.array_split(test_idx, splits) if len(c)]
    folds = []
    for c in chunks:
        tr = np.arange(0, int(c[0]))
        if len(tr) == 0:
            continue
        folds.append((arr[tr], arr[c]))
    return folds


def _daily_bin_series(sub: pd.DataFrame, q_lo: float, q_hi: float, response_col: str) -> pd.Series:
    # Last band is closed on 1.0.
    if q_hi >= 1.0 - 1e-12:
        z = sub[(sub["q_psi"] >= q_lo) & (sub["q_psi"] <= q_hi)]
    else:
        z = sub[(sub["q_psi"] >= q_lo) & (sub["q_psi"] < q_hi)]
    if z.empty:
        return pd.Series(dtype="float64")
    return z.groupby("date", observed=True)[response_col].mean().sort_index()


def _moving_block_bootstrap_mean(values: np.ndarray, reps: int, seed: int, block: int | None = None) -> tuple[float,float,float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    est = float(np.mean(x))
    if n == 1:
        return est, est, 1.0 if est >= 0 else 0.5
    if block is None:
        block = max(5, int(round(n ** (1/3))))
    block = min(block, n)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    starts_max = max(1, n - block + 1)
    need_blocks = int(math.ceil(n / block))
    for r in range(reps):
        pieces = []
        starts = rng.integers(0, starts_max, size=need_blocks)
        for s in starts:
            pieces.append(x[s:s+block])
        sample = np.concatenate(pieces)[:n]
        means[r] = sample.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    # one-sided p for H1 mean < 0, with +1 correction
    p_lt0 = float((1 + np.sum(means >= 0.0)) / (reps + 1))
    return float(lo), float(hi), p_lt0


def certify_candidate(
    frame: pd.DataFrame,
    *,
    universe: str,
    q0: float,
    horizons: Sequence[int],
    return_modes: Sequence[str] = ("market_residual","raw"),
    splits: int = 4,
    min_train_frac: float = 0.5,
    bootstrap_reps: int = 1000,
    alpha: float = 0.05,
    min_effect_sigma: float = 0.02,
    min_obs_per_bin: int = 250,
    min_dates_per_bin: int = 40,
    min_train_dates_per_bin: int = 40,
    seed: int = 20260812,
) -> pd.DataFrame:
    rows: list[dict] = []
    mid = (q0 + 1.0) / 2.0
    bins = [(q0, mid), (mid, 1.0)]

    for mode in return_modes:
        md = frame[frame["return_mode"].astype(str) == mode]
        if md.empty:
            continue
        dates = sorted(md["date"].dropna().unique())
        folds = expanding_walkforward_dates(dates, splits=splits, min_train_frac=min_train_frac)
        for fold_i, (train_dates, test_dates) in enumerate(folds, 1):
            train_set = set(pd.to_datetime(train_dates))
            test_set = set(pd.to_datetime(test_dates))
            fold_indices: list[int] = []
            fold_pvals: list[float] = []
            for h in horizons:
                hs = md[md["horizon"] == int(h)]
                tr = hs[hs["date"].isin(train_set)]
                te = hs[hs["date"].isin(test_set)]
                bin_results = []
                eligible = True
                for bidx, (lo, hi) in enumerate(bins):
                    tr_daily = _daily_bin_series(tr, lo, hi, "phase_product_robust")
                    te_daily = _daily_bin_series(te, lo, hi, "phase_product_robust")
                    # raw observation count is a guardrail only, not the estimand.
                    if hi >= 1.0 - 1e-12:
                        te_rows = te[(te["q_psi"] >= lo) & (te["q_psi"] <= hi)]
                    else:
                        te_rows = te[(te["q_psi"] >= lo) & (te["q_psi"] < hi)]
                    if len(te_rows) < min_obs_per_bin or len(te_daily) < min_dates_per_bin or len(tr_daily) < min_train_dates_per_bin:
                        eligible = False
                    sigma_train = float(tr_daily.std(ddof=1)) if len(tr_daily) > 1 else np.nan
                    epsilon = min_effect_sigma * sigma_train if np.isfinite(sigma_train) else np.nan
                    est = float(te_daily.mean()) if len(te_daily) else np.nan
                    ci_lo, ci_hi, p = _moving_block_bootstrap_mean(
                        te_daily.to_numpy(), bootstrap_reps,
                        seed + 100000*fold_i + 1000*int(h) + 10*bidx + int(round(q0*100)),
                    )
                    effect_pass = bool(np.isfinite(est) and np.isfinite(epsilon) and est <= -epsilon)
                    ci_pass = bool(np.isfinite(ci_hi) and ci_hi < 0.0)
                    bin_results.append({
                        "bin": bidx, "q_lo": lo, "q_hi": hi, "estimate": est,
                        "ci_lo": ci_lo, "ci_hi": ci_hi, "p_lt0": p,
                        "sigma_train_daily": sigma_train, "epsilon": epsilon,
                        "effect_floor_pass": effect_pass, "ci_negative_pass": ci_pass,
                        "n_test_rows": int(len(te_rows)), "n_test_dates": int(len(te_daily)),
                        "n_train_dates": int(len(tr_daily)),
                    })
                p_iut = float(max((b["p_lt0"] for b in bin_results), default=1.0))
                tail_negative = bool(eligible and all(b["effect_floor_pass"] and b["ci_negative_pass"] for b in bin_results))
                row = {
                    "universe": universe, "return_mode": mode, "fold": fold_i, "horizon": int(h),
                    "q0": float(q0), "eligible": eligible, "tail_negative_local": tail_negative,
                    "candidate_p_iut": p_iut, "passes_holm": False, "certified_tail_negative_fold": False,
                    "tail_bins_json": json.dumps(bin_results, separators=(",", ":")),
                    "test_start": str(pd.Timestamp(min(test_dates)).date()) if len(test_dates) else "",
                    "test_end": str(pd.Timestamp(max(test_dates)).date()) if len(test_dates) else "",
                }
                rows.append(row)
                fold_indices.append(len(rows)-1)
                fold_pvals.append(p_iut)
            holm = _bh_holm(fold_pvals, alpha)
            for idx, hp in zip(fold_indices, holm):
                rows[idx]["passes_holm"] = bool(hp)
                rows[idx]["certified_tail_negative_fold"] = bool(rows[idx]["tail_negative_local"] and hp)
    return pd.DataFrame(rows)


def aggregate_regimes(folds: pd.DataFrame, min_folds: int = 3, min_fold_fraction: float = 0.75) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    rows=[]
    for keys, g in folds.groupby(["universe","return_mode","q0","horizon"], observed=True):
        universe, mode, q0, h = keys
        eligible = int(g["eligible"].sum())
        support = int(g["certified_tail_negative_fold"].sum())
        denom = max(1, eligible)
        frac = support/denom
        certified = support >= min_folds and frac >= min_fold_fraction
        rows.append({"universe":universe,"return_mode":mode,"q0":float(q0),"horizon":int(h),
                     "eligible_folds":eligible,"certified_folds":support,"support_fraction":frac,
                     "certified_tail_negative":bool(certified)})
    return pd.DataFrame(rows)


def identify_hc(regimes: pd.DataFrame, horizons: Sequence[int], min_consecutive: int = 3, min_tail_fraction: float = 0.8) -> list[dict]:
    out=[]
    H=list(map(int,horizons))
    if regimes.empty:
        return out
    for (universe, mode, q0), g in regimes.groupby(["universe","return_mode","q0"], observed=True):
        mp={int(r.horizon):bool(r.certified_tail_negative) for r in g.itertuples()}
        identified=False; hc=None
        for i,h in enumerate(H):
            tail=H[i:]
            first=tail[:min_consecutive]
            if len(first)<min_consecutive:
                continue
            if not all(mp.get(x,False) for x in first):
                continue
            frac=sum(mp.get(x,False) for x in tail)/len(tail)
            if frac >= min_tail_fraction:
                identified=True; hc=h; break
        out.append({"universe":universe,"return_mode":mode,"q0":float(q0),"identified":identified,
                    "hc_grid":int(hc) if hc is not None else None})
    return out


def consensus_for_q0(hc_rows: list[dict], universe: str, q0: float) -> dict:
    rows=[r for r in hc_rows if r["universe"]==universe and abs(r["q0"]-q0)<1e-12]
    by_mode={r["return_mode"]:r for r in rows}
    needed=["market_residual","raw"]
    if not all(m in by_mode and by_mode[m]["identified"] for m in needed):
        return {"identified":False}
    vals=[int(by_mode[m]["hc_grid"]) for m in needed]
    return {"identified":True,"hc_grid":max(vals),"mode_hc":{m:int(by_mode[m]["hc_grid"]) for m in needed}}


def select_transport_candidate(candidate_summaries: list[dict]) -> dict:
    eligible=[]
    for c in candidate_summaries:
        if c["discovery_consensus"].get("identified") and c["second_consensus"].get("identified"):
            delta=abs(c["discovery_consensus"]["hc_grid"]-c["second_consensus"]["hc_grid"])
            eligible.append((delta, c["q0"], c))
    if not eligible:
        return {"selected":False,"reason":"no_robust_tail_candidate_identified_in_both_development_universes_and_both_modes"}
    eligible.sort(key=lambda x:(x[0], x[1]))  # smallest delta, then broadest tail (lowest q0)
    c=eligible[0][2]
    return {"selected":True,"normalization":"global_percentile","response":"cross_sectional_huber_clipped_phase_product",
            "huber_c":HUBER_C,"q0":c["q0"],"discovery_hc_grid":c["discovery_consensus"]["hc_grid"],
            "second_hc_grid":c["second_consensus"]["hc_grid"],
            "development_hc_delta_days":abs(c["discovery_consensus"]["hc_grid"]-c["second_consensus"]["hc_grid"])}


def extreme_attribution(frame: pd.DataFrame, universe: str, top_n: int = 500) -> pd.DataFrame:
    x=frame.copy()
    x["robustization_delta"]=(x["phase_product"]-x["phase_product_robust"]).abs()
    x=x.nlargest(top_n,"robustization_delta")
    x.insert(0,"universe",universe)
    return x[["universe","date","ticker","return_mode","horizon","q_psi","phase_product","phase_product_robust","robust_scale_xs","was_clipped","robustization_delta"]]


def run_exploratory(
    discovery_panel: Path, second_panel: Path, out_dir: Path, horizons: Sequence[int], q0_grid: Sequence[float],
    bootstrap_reps: int=1000, alpha: float=0.05, min_effect_sigma: float=0.02,
    min_folds: int=3, min_fold_fraction: float=0.75, hc_min_consecutive: int=3,
    hc_min_tail_fraction: float=0.8, walkforward_splits: int=4, min_train_frac: float=0.5,
    seed: int=20260812,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds=[]; all_regs=[]; hc_rows=[]; candidate_summaries=[]
    prepared={}
    for label,path in [("discovery",discovery_panel),("second_universe",second_panel)]:
        raw=load_panel_streaming(path,horizons)
        have=sorted(map(int,raw["horizon"].unique()))
        missing=[int(h) for h in horizons if int(h) not in have]
        if missing:
            raise ValueError(f"{label} panel missing requested horizons: {missing}. Rebuild/extend panel before v9.2.")
        pre=add_global_percentile_and_robust_response(raw)
        prepared[label]=pre
        extreme_attribution(pre,label).to_csv(out_dir/f"{label}_robustization_extremes.csv",index=False)
        del raw
        for q0 in q0_grid:
            f=certify_candidate(pre, universe=label, q0=float(q0), horizons=horizons,
                                splits=walkforward_splits,min_train_frac=min_train_frac,
                                bootstrap_reps=bootstrap_reps,alpha=alpha,min_effect_sigma=min_effect_sigma,seed=seed)
            r=aggregate_regimes(f,min_folds=min_folds,min_fold_fraction=min_fold_fraction)
            all_folds.append(f); all_regs.append(r)
    folds=pd.concat(all_folds,ignore_index=True)
    regs=pd.concat(all_regs,ignore_index=True)
    folds.to_csv(out_dir/"robust_tail_fold_certification.csv",index=False)
    regs.to_csv(out_dir/"robust_tail_regimes.csv",index=False)
    hc_rows=identify_hc(regs,horizons,min_consecutive=hc_min_consecutive,min_tail_fraction=hc_min_tail_fraction)
    pd.DataFrame(hc_rows).to_csv(out_dir/"robust_tail_hc.csv",index=False)
    for q0 in q0_grid:
        candidate_summaries.append({"q0":float(q0),
            "discovery_consensus":consensus_for_q0(hc_rows,"discovery",float(q0)),
            "second_consensus":consensus_for_q0(hc_rows,"second_universe",float(q0))})
    selected=select_transport_candidate(candidate_summaries)
    (out_dir/"selected_transport_hypothesis_v9_2.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    summary={
        "status":"EXPLORATORY_TRANSPORT_V9_2_ROBUST_DELAYED",
        "scientific_change":"new development-only estimand: cross-sectional Huber-clipped phase_product (median/MAD, c=1.345) before daily percentile-tail averaging; intended to prevent isolated extreme corporate-action-like observations from dominating the phase mean",
        "confirmatory_v8_v9_unchanged":True,
        "normalization":"global empirical percentile within universe x date x return_mode x horizon",
        "response":"median + clip(phase_product-median, +/- 1.345*1.4826*MAD) within date x return_mode x horizon",
        "horizons":list(map(int,horizons)),"q0_grid":list(map(float,q0_grid)),
        "candidate_summaries":candidate_summaries,"selected_transport_hypothesis":selected,
        "rules":{"bootstrap_reps":bootstrap_reps,"alpha":alpha,"min_effect_sigma":min_effect_sigma,
                 "effect_scale":"train SD of daily robust tail-bin means","tail_bins":2,
                 "min_folds":min_folds,"min_fold_fraction":min_fold_fraction,
                 "hc_min_consecutive":hc_min_consecutive,"hc_min_tail_fraction":hc_min_tail_fraction,
                 "walkforward_splits":walkforward_splits,"min_train_frac":min_train_frac,"seed":seed,
                 "huber_c":HUBER_C,"mad_normalizer":MAD_NORMALIZER},
        "guardrail":"Discovery and second universe are development data. Do not inspect a third universe until a v9.2 protocol is frozen and bound to its ticker list."
    }
    summary["summary_sha256"]=canonical_json_sha256(summary)
    (out_dir/"transport_exploratory_summary_v9_2.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

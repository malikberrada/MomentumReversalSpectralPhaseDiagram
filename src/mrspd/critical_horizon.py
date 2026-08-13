from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .phase_surface import (
    PhaseSurfaceConfig,
    SplineSurface,
    _balanced_fit_sample,
    _date_weights,
    _roots_from_curve,
    _single_phase_fold_certification,
    _walkforward_date_blocks,
)


DEFAULT_CRITICAL_HORIZONS: tuple[int, ...] = (
    63, 72, 84, 96, 105, 112, 119, 126, 133, 140, 147, 168
)


@dataclass(frozen=True)
class CriticalHorizonConfig:
    spline_knots: int = 7
    bootstrap_reps: int = 1000
    max_fit_rows: int = 250_000
    cert_alpha: float = 0.05
    cert_min_effect_sigma: float = 0.02
    cert_abs_effect: float = 0.0
    cert_min_obs_per_band: int = 250
    cert_min_dates_per_band: int = 40
    cert_min_folds: int = 3
    cert_min_fold_fraction: float = 0.75
    cert_single_phase_bins: int = 5
    root_grid_size: int = 2048
    root_domain_qlo: float = 0.005
    root_domain_qhi: float = 0.995
    hc_min_consecutive: int = 3
    hc_min_tail_fraction: float = 0.80
    random_seed: int = 20260812


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if len(finite_idx) == 0:
        return out
    vals = p[finite_idx]
    order = np.argsort(vals)
    m = len(vals)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, j in enumerate(order):
        candidate = (m - rank) * vals[j]
        running = max(running, candidate)
        adj_sorted[rank] = min(1.0, running)
    # adj_sorted is stored in order-rank coordinates; map back.
    for rank, j in enumerate(order):
        out[finite_idx[j]] = adj_sorted[rank]
    return out


def _apply_horizon_family_holm(folds: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Correct the dense horizon scan within each return-mode x fold family."""
    if folds.empty:
        return folds.copy()
    z = folds.copy()
    z["p_holm_horizon_family"] = np.nan
    z["passes_horizon_holm"] = False
    z["certified_r_only_fold"] = False
    z["certified_m_only_fold"] = False

    for (_, _), idx in z.groupby(["return_mode", "fold"], sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        p = z.loc[loc, "candidate_p_iut"].to_numpy(dtype=float)
        adj = _holm_adjust(p)
        z.loc[loc, "p_holm_horizon_family"] = adj
        passed = np.isfinite(adj) & (adj <= float(alpha))
        z.loc[loc, "passes_horizon_holm"] = passed

    z["certified_r_only_fold"] = (
        z["passes_horizon_holm"].astype(bool)
        & z["phase"].astype(str).eq("R")
        & z["eligible"].astype(bool)
    )
    z["certified_m_only_fold"] = (
        z["passes_horizon_holm"].astype(bool)
        & z["phase"].astype(str).eq("M")
        & z["eligible"].astype(bool)
    )
    return z


def _aggregate_horizon_regimes(
    folds: pd.DataFrame,
    *,
    min_folds: int,
    min_fold_fraction: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    if folds.empty:
        return pd.DataFrame(rows)
    for (mode, h), g in folds.groupby(["return_mode", "horizon"], sort=True):
        total = int(g["fold"].nunique())
        required = max(int(min_folds), int(math.ceil(float(min_fold_fraction) * total)))
        r_support = int(g["certified_r_only_fold"].sum())
        m_support = int(g["certified_m_only_fold"].sum())
        root_free = int((g["train_n_roots"] == 0).sum())
        if r_support >= required and m_support >= required:
            status = "CONFLICT"
        elif r_support >= required:
            status = "R"
        elif m_support >= required:
            status = "M"
        else:
            status = "UNRESOLVED"
        rows.append({
            "return_mode": mode,
            "horizon": int(h),
            "folds_total": total,
            "min_folds_required": required,
            "r_folds_support": r_support,
            "r_support_fraction": r_support / float(total) if total else np.nan,
            "m_folds_support": m_support,
            "m_support_fraction": m_support / float(total) if total else np.nan,
            "root_free_train_folds": root_free,
            "root_free_train_fraction": root_free / float(total) if total else np.nan,
            "certified_regime": status,
            "certified_r_only": bool(status == "R"),
            "certified_m_only": bool(status == "M"),
        })
    return pd.DataFrame(rows)


def _find_onset(
    horizons: Sequence[int],
    r_flags: Sequence[bool],
    m_flags: Sequence[bool] | None = None,
    *,
    min_consecutive: int = 3,
    min_tail_fraction: float = 0.80,
) -> dict:
    """Find first persistent R-only onset on a discrete horizon grid.

    Primary onset requires:
      * the current horizon and the next ``min_consecutive-1`` scanned horizons
        are certified R-only;
      * no later scanned horizon is certified M-only;
      * at least ``min_tail_fraction`` of the remaining grid is R-only.

    This returns a *grid bracket*, not a continuous confidence interval.
    """
    hs = np.asarray(horizons, dtype=int)
    r = np.asarray(r_flags, dtype=bool)
    m = np.zeros(len(r), dtype=bool) if m_flags is None else np.asarray(m_flags, dtype=bool)
    if len(hs) != len(r) or len(hs) != len(m):
        raise ValueError("horizons/r_flags/m_flags must have equal length")
    empty_result = {
        "identified": False,
        "reason": "",
        "hc_grid": None,
        "hc_lower_exclusive": None,
        "hc_upper_inclusive": None,
        "hc_midpoint": None,
        "hc_months_approx": None,
        "min_consecutive": max(1, int(min_consecutive)),
        "tail_r_fraction": None,
        "strict_r_tail": False,
        "later_m_certified": None,
        "n_tail_horizons": 0,
    }
    if len(hs) == 0:
        return {**empty_result, "reason": "empty_grid"}
    order = np.argsort(hs)
    hs, r, m = hs[order], r[order], m[order]
    k = max(1, int(min_consecutive))
    for i in range(len(hs)):
        if i + k > len(hs):
            break
        if not bool(np.all(r[i:i + k])):
            continue
        tail_r = r[i:]
        tail_m = m[i:]
        tail_fraction = float(np.mean(tail_r))
        if bool(np.any(tail_m)):
            continue
        if tail_fraction + 1e-15 < float(min_tail_fraction):
            continue
        strict_tail = bool(np.all(tail_r))
        lower = int(hs[i - 1]) if i > 0 else None
        upper = int(hs[i])
        return {
            "identified": True,
            "hc_grid": upper,
            "hc_lower_exclusive": lower,
            "hc_upper_inclusive": upper,
            "hc_midpoint": (0.5 * (lower + upper)) if lower is not None else float(upper),
            "hc_months_approx": float(upper / 21.0),
            "min_consecutive": k,
            "tail_r_fraction": tail_fraction,
            "strict_r_tail": strict_tail,
            "later_m_certified": False,
            "n_tail_horizons": int(len(tail_r)),
        }
    return {
        **empty_result,
        "reason": "no_persistent_r_only_onset",
        "min_consecutive": k,
        "min_tail_fraction": float(min_tail_fraction),
    }


def _estimate_hc_tables(
    fold_table: pd.DataFrame,
    regime_table: pd.DataFrame,
    *,
    min_consecutive: int,
    min_tail_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mode_rows: list[dict] = []
    fold_rows: list[dict] = []

    for mode, g in regime_table.groupby("return_mode", sort=False):
        g = g.sort_values("horizon")
        est = _find_onset(
            g["horizon"].to_numpy(int),
            g["certified_r_only"].to_numpy(bool),
            g["certified_m_only"].to_numpy(bool),
            min_consecutive=min_consecutive,
            min_tail_fraction=min_tail_fraction,
        )
        mode_rows.append({"return_mode": mode, **est})

    for (mode, fold), g in fold_table.groupby(["return_mode", "fold"], sort=False):
        g = g.sort_values("horizon")
        est = _find_onset(
            g["horizon"].to_numpy(int),
            g["certified_r_only_fold"].to_numpy(bool),
            g["certified_m_only_fold"].to_numpy(bool),
            min_consecutive=min_consecutive,
            min_tail_fraction=min_tail_fraction,
        )
        fold_rows.append({"return_mode": mode, "fold": int(fold), **est})

    mode_df = pd.DataFrame(mode_rows)
    fold_df = pd.DataFrame(fold_rows)

    consensus: dict = {"identified": False}
    good = mode_df[mode_df.get("identified", False).astype(bool)] if not mode_df.empty else pd.DataFrame()
    if not good.empty and set(regime_table["return_mode"].unique()).issubset(set(good["return_mode"])):
        upper = int(good["hc_upper_inclusive"].max())
        lowers = good["hc_lower_exclusive"].dropna()
        lower = int(lowers.max()) if len(lowers) else None
        consensus = {
            "identified": True,
            "definition": "first bracket by which every return mode has entered its persistent certified R-only tail",
            "hc_lower_exclusive": lower,
            "hc_upper_inclusive": upper,
            "hc_grid": upper,
            "hc_midpoint": (0.5 * (lower + upper)) if lower is not None else float(upper),
            "hc_months_approx": float(upper / 21.0),
            "modes": good["return_mode"].tolist(),
        }
    return mode_df, fold_df, consensus


def _plot_scan(regimes: pd.DataFrame, estimates: pd.DataFrame, out_dir: Path) -> None:
    if regimes.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for mode, g in regimes.groupby("return_mode", sort=False):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["r_support_fraction"], marker="o", label=str(mode))
        if not estimates.empty:
            q = estimates[(estimates["return_mode"] == mode) & estimates["identified"].astype(bool)]
            if not q.empty:
                ax.axvline(float(q.iloc[0]["hc_grid"]), linestyle="--", alpha=0.6)
    ax.axhline(0.75, linestyle=":", alpha=0.7, label="3/4 fold support")
    ax.set_xlabel("Horizon h (trading days)")
    ax.set_ylabel("Fraction of folds certified R-only after horizon-family Holm")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("MRSPD critical-horizon scan")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "critical_horizon_scan.png", dpi=180)
    plt.close(fig)


def run_critical_horizon_scan(
    panel: pd.DataFrame,
    cfg,
    out_dir: Path,
    *,
    horizons: Sequence[int] | None = None,
    spline_knots: int = 7,
    bootstrap_reps: int = 1000,
    max_fit_rows: int = 250_000,
    cert_alpha: float = 0.05,
    cert_min_effect_sigma: float = 0.02,
    cert_abs_effect: float = 0.0,
    cert_min_obs_per_band: int = 250,
    cert_min_dates_per_band: int = 40,
    cert_min_folds: int = 3,
    cert_min_fold_fraction: float = 0.75,
    cert_single_phase_bins: int = 5,
    hc_min_consecutive: int = 3,
    hc_min_tail_fraction: float = 0.80,
) -> dict:
    """Estimate the discrete critical horizon at which R-only becomes persistent.

    This is intentionally narrower than ``run_phase_surface_analysis``.  It fits
    only the continuous tensor-product surface needed to determine whether the
    training slice has candidate roots, then reuses the v5 OOS single-phase
    certification.  Dense-horizon multiplicity is controlled with Holm within
    each return-mode x fold before cross-fold aggregation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required_cols = ["date", "return_mode", "horizon", "psi_primary", "phase_product"]
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"Critical-horizon panel is missing columns {missing_cols}")
    available = sorted(int(x) for x in panel["horizon"].dropna().unique())
    requested = sorted(set(int(x) for x in (horizons if horizons is not None else available)))
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            f"Panel is missing critical-scan horizons {missing}. Rebuild the panel with --horizons "
            + ",".join(map(str, requested))
        )
    # One compact ownership copy only. Older code copied the full-width panel
    # and then copied the horizon subset again, causing multi-GiB peaks.
    panel = panel.loc[panel["horizon"].isin(requested), required_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(panel["date"].dtype):
        panel["date"] = pd.to_datetime(panel["date"])
    if len(requested) < 3:
        raise ValueError("Critical-horizon scan needs at least three horizons")

    ccfg = CriticalHorizonConfig(
        spline_knots=spline_knots,
        bootstrap_reps=bootstrap_reps,
        max_fit_rows=max_fit_rows,
        cert_alpha=cert_alpha,
        cert_min_effect_sigma=cert_min_effect_sigma,
        cert_abs_effect=cert_abs_effect,
        cert_min_obs_per_band=cert_min_obs_per_band,
        cert_min_dates_per_band=cert_min_dates_per_band,
        cert_min_folds=cert_min_folds,
        cert_min_fold_fraction=cert_min_fold_fraction,
        cert_single_phase_bins=cert_single_phase_bins,
        hc_min_consecutive=hc_min_consecutive,
        hc_min_tail_fraction=hc_min_tail_fraction,
        random_seed=int(cfg.random_seed),
    )
    scfg = PhaseSurfaceConfig(
        spline_knots=ccfg.spline_knots,
        bootstrap_reps=ccfg.bootstrap_reps,
        max_fit_rows=ccfg.max_fit_rows,
        random_seed=ccfg.random_seed,
        cert_alpha=ccfg.cert_alpha,
        cert_min_effect_sigma=ccfg.cert_min_effect_sigma,
        cert_abs_effect=ccfg.cert_abs_effect,
        cert_min_obs_per_side=ccfg.cert_min_obs_per_band,
        cert_min_dates_per_side=ccfg.cert_min_dates_per_band,
        cert_min_folds=ccfg.cert_min_folds,
        cert_min_fold_fraction=ccfg.cert_min_fold_fraction,
        cert_single_phase_bins=ccfg.cert_single_phase_bins,
    )

    fold_rows: list[dict] = []
    max_h = max(requested)
    block_len = max(1, int(math.ceil(max_h / int(cfg.anchor_stride))))

    for mode_i, (mode, pm) in enumerate(panel.groupby("return_mode", sort=False)):
        blocks = _walkforward_date_blocks(pm, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, start=1):
            purge = pd.offsets.BDay(max_h)
            train = pm[pm["date"] < (test_start - purge)].copy()
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)].copy()
            if len(train) < 1000 or len(test) < 200:
                continue

            fit_seed = ccfg.random_seed + 100_000 * (mode_i + 1) + fold
            fit_train = _balanced_fit_sample(train, ccfg.max_fit_rows, fit_seed)
            surface = SplineSurface(
                task="continuous",
                interaction=True,
                context=False,
                n_knots=ccfg.spline_knots,
                degree=3,
                ridge_alpha=4.0,
                logistic_c=0.5,
            ).fit(fit_train, _date_weights(fit_train))

            lo, hi = np.nanquantile(
                train["psi_primary"].to_numpy(float),
                [ccfg.root_domain_qlo, ccfg.root_domain_qhi],
            )
            psi_grid = np.linspace(lo, hi, ccfg.root_grid_size)

            for h in requested:
                train_h = train[train["horizon"] == h]
                test_h = test[test["horizon"] == h]
                if len(train_h) < 200 or len(test_h) < 100:
                    fold_rows.append({
                        "return_mode": mode,
                        "fold": fold,
                        "test_start": test_start,
                        "test_end": test_end,
                        "horizon": int(h),
                        "train_n_roots": np.nan,
                        "eligible": False,
                        "reason": "insufficient_horizon_rows",
                        "certified_single_phase": False,
                    })
                    continue
                eval_df = pd.DataFrame({
                    "psi_primary": psi_grid,
                    "horizon": np.full(len(psi_grid), h, dtype=float),
                })
                ghat = surface.predict(eval_df)
                roots = _roots_from_curve(psi_grid, ghat)
                single = _single_phase_fold_certification(
                    train_h,
                    test_h,
                    train_n_roots=len(roots),
                    block_len=block_len,
                    scfg=scfg,
                    seed=fit_seed + 50_000 * int(h),
                )
                fold_rows.append({
                    "return_mode": mode,
                    "fold": fold,
                    "test_start": test_start,
                    "test_end": test_end,
                    "horizon": int(h),
                    "train_n_roots": int(len(roots)),
                    "train_phase_path": _phase_path_local(psi_grid, ghat, roots),
                    **single,
                })
            print(f"critical-horizon {mode} fold {fold}/{len(blocks)} complete")

    folds = pd.DataFrame(fold_rows)
    if "candidate_p_iut" not in folds.columns:
        folds["candidate_p_iut"] = np.nan
    if "phase" not in folds.columns:
        folds["phase"] = ""
    if "eligible" not in folds.columns:
        folds["eligible"] = False
    folds = _apply_horizon_family_holm(folds, ccfg.cert_alpha)
    regimes = _aggregate_horizon_regimes(
        folds,
        min_folds=ccfg.cert_min_folds,
        min_fold_fraction=ccfg.cert_min_fold_fraction,
    )
    mode_hc, fold_hc, consensus = _estimate_hc_tables(
        folds,
        regimes,
        min_consecutive=ccfg.hc_min_consecutive,
        min_tail_fraction=ccfg.hc_min_tail_fraction,
    )

    folds.to_csv(out_dir / "critical_horizon_fold_certification.csv", index=False)
    regimes.to_csv(out_dir / "critical_horizon_regimes.csv", index=False)
    mode_hc.to_csv(out_dir / "critical_horizon_estimates.csv", index=False)
    fold_hc.to_csv(out_dir / "critical_horizon_fold_estimates.csv", index=False)
    _plot_scan(regimes, mode_hc, out_dir)

    fold_stats: list[dict] = []
    if not fold_hc.empty and "identified" in fold_hc:
        for mode, g in fold_hc.groupby("return_mode", sort=False):
            q = g[g["identified"].astype(bool) & g["hc_grid"].notna()]
            fold_stats.append({
                "return_mode": mode,
                "folds_identified": int(len(q)),
                "folds_total": int(g["fold"].nunique()),
                "median_hc_grid": float(q["hc_grid"].median()) if len(q) else None,
                "iqr_hc_grid": float(q["hc_grid"].quantile(0.75) - q["hc_grid"].quantile(0.25)) if len(q) else None,
            })

    summary = {
        "definition": "h_c is the first scanned horizon entering a persistent, OOS-certified R-only tail",
        "scan_horizons": requested,
        "multiplicity": "Holm-Bonferroni across scanned horizons within each return-mode x fold",
        "single_phase_rule": "root-free training slice + all OOS Psi bands same-sign with effect floor and block-bootstrap CI",
        "min_consecutive_r_horizons": ccfg.hc_min_consecutive,
        "min_r_tail_fraction": ccfg.hc_min_tail_fraction,
        "cert_alpha": ccfg.cert_alpha,
        "cert_min_effect_sigma": ccfg.cert_min_effect_sigma,
        "cert_min_folds": ccfg.cert_min_folds,
        "cert_min_fold_fraction": ccfg.cert_min_fold_fraction,
        "per_mode_hc": mode_hc.to_dict("records"),
        "fold_hc_stability": fold_stats,
        "consensus_hc": consensus,
        "interpretation_guardrail": "The h_c interval is a discrete scan-grid bracket, not a continuous confidence interval.",
    }
    (out_dir / "critical_horizon_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _phase_path_local(grid: np.ndarray, g: np.ndarray, roots: list[dict]) -> str:
    if len(grid) == 0:
        return ""
    cuts = [float(grid[0])] + [float(r["psi_star"]) for r in roots] + [float(grid[-1])]
    states: list[str] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = 0.5 * (a + b)
        val = float(np.interp(mid, grid, g))
        states.append("M" if val > 0.0 else "R" if val < 0.0 else "0")
    return "→".join(states)

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .critical_horizon import DEFAULT_CRITICAL_HORIZONS, _phase_path_local
from .phase_surface import (
    PhaseSurfaceConfig,
    SplineSurface,
    _balanced_fit_sample,
    _date_weights,
    _roots_from_curve,
    _single_phase_fold_certification,
    _walkforward_date_blocks,
)


DEFAULT_REFINEMENT_LOWER_EXCLUSIVE = 119
DEFAULT_REFINEMENT_UPPER_INCLUSIVE = 133
DEFAULT_REFINEMENT_HORIZONS: tuple[int, ...] = tuple(range(120, 134))
DEFAULT_REFINEMENT_FIT_HORIZONS: tuple[int, ...] = tuple(
    sorted(set(DEFAULT_CRITICAL_HORIZONS).union(DEFAULT_REFINEMENT_HORIZONS))
)


@dataclass(frozen=True)
class RefinementConfig:
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


def _load_coarse_evidence(coarse_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    coarse_dir = Path(coarse_dir)
    summary_path = coarse_dir / "critical_horizon_summary.json"
    regimes_path = coarse_dir / "critical_horizon_regimes.csv"
    folds_path = coarse_dir / "critical_horizon_fold_certification.csv"
    missing = [p.name for p in (summary_path, regimes_path, folds_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Expected v6 coarse evidence in coarse_dir; missing: " + ", ".join(missing)
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    regimes = pd.read_csv(regimes_path)
    folds = pd.read_csv(folds_path, parse_dates=["test_start", "test_end"])
    return summary, regimes, folds


def _coarse_tail_guard(
    coarse_regimes: pd.DataFrame,
    *,
    return_mode: str,
    upper: int,
) -> dict:
    g = coarse_regimes[coarse_regimes["return_mode"].astype(str) == str(return_mode)].copy()
    g = g.sort_values("horizon")
    tail = g[g["horizon"] >= int(upper)].copy()
    if tail.empty:
        return {
            "ok": False,
            "reason": "no_coarse_tail_at_or_above_refinement_upper",
            "n_tail": 0,
            "tail_r_fraction": np.nan,
            "later_m": False,
        }
    states = tail["certified_regime"].astype(str)
    r_fraction = float(states.eq("R").mean())
    later_m = bool(states.eq("M").any())
    ok = (r_fraction >= 0.80) and not later_m and bool(states.iloc[-1] == "R")
    return {
        "ok": bool(ok),
        "reason": "" if ok else "coarse_tail_not_persistently_r_only",
        "n_tail": int(len(tail)),
        "tail_r_fraction": r_fraction,
        "later_m": later_m,
    }


def _v6_fit_seed(random_seed: int, mode_index: int, fold: int) -> int:
    """Exactly reproduce the seed scheme used by the v6 coarse scan."""
    return int(random_seed) + 100_000 * (int(mode_index) + 1) + int(fold)


def _coarse_fit_subset(train: pd.DataFrame, coarse_horizons: Sequence[int]) -> pd.DataFrame:
    """Freeze the refinement surface fit to the exact coarse-horizon design.

    Dense refinement rows are measurement rows only. They must never alter the
    training distribution of the surface that determines the root-free gate.
    """
    hs = {int(x) for x in coarse_horizons}
    return train[train["horizon"].astype(int).isin(hs)].copy()


def _required_fold_support(total: int, min_folds: int, min_fold_fraction: float) -> int:
    return max(int(min_folds), int(math.ceil(float(min_fold_fraction) * int(total))))


def _fixed_sequence_gatekeeping(
    fine_folds: pd.DataFrame,
    coarse_folds: pd.DataFrame,
    *,
    upper_anchor: int,
    alpha: float,
) -> pd.DataFrame:
    """Ordered refinement from a v6-certified coarse upper anchor.

    The bracket was discovered before the fine scan. For each return-mode x
    fold, the already-tested v6 upper anchor is the gate. Starting one day below
    the anchor, hypotheses are evaluated in the pre-specified order h=upper-1,
    upper-2, ... . Testing stops after the first local failure. Fixed-sequence
    gatekeeping controls FWER at alpha under arbitrary dependence without the
    artificial power loss caused by treating 14 nearly identical horizons as an
    unordered Holm family.

    Local certification itself is unchanged: same 5 Psi bands, effect floor,
    block bootstrap CI, root-free train gate, and alpha.
    """
    z = fine_folds.copy()
    if z.empty:
        return z
    for col, default in [
        ("candidate_p_iut", np.nan),
        ("phase", ""),
        ("eligible", False),
        ("certified_single_phase", False),
    ]:
        if col not in z.columns:
            z[col] = default
    z["local_r_pass"] = (
        z["eligible"].astype(bool)
        & z["certified_single_phase"].astype(bool)
        & z["phase"].astype(str).eq("R")
        & np.isfinite(z["candidate_p_iut"].to_numpy(dtype=float))
        & (z["candidate_p_iut"].to_numpy(dtype=float) <= float(alpha))
    )
    z["fixed_sequence_r_pass"] = False
    z["fixed_sequence_active"] = False
    z["gate_source"] = ""

    for (mode, fold), idx in z.groupby(["return_mode", "fold"], sort=False).groups.items():
        loc = list(idx)
        cf = coarse_folds[
            (coarse_folds["return_mode"].astype(str) == str(mode))
            & (coarse_folds["fold"].astype(int) == int(fold))
            & (coarse_folds["horizon"].astype(int) == int(upper_anchor))
        ]
        anchor_pass = False
        if not cf.empty:
            r = cf.iloc[0]
            if "certified_r_only_fold" in cf.columns:
                anchor_pass = bool(r["certified_r_only_fold"])
            else:
                anchor_pass = bool(
                    bool(r.get("eligible", False))
                    and str(r.get("phase", "")) == "R"
                    and bool(r.get("passes_horizon_holm", False))
                )

        rows = z.loc[loc].sort_values("horizon", ascending=False)
        active = bool(anchor_pass)
        for ridx, row in rows.iterrows():
            h = int(row["horizon"])
            if h == int(upper_anchor):
                # The upper point is not re-discovered by the dense scan. Its
                # gate status is inherited from the exact v6 coarse evidence.
                z.at[ridx, "fixed_sequence_active"] = bool(anchor_pass)
                z.at[ridx, "fixed_sequence_r_pass"] = bool(anchor_pass)
                z.at[ridx, "gate_source"] = "v6_coarse_anchor"
                continue
            z.at[ridx, "fixed_sequence_active"] = bool(active)
            z.at[ridx, "gate_source"] = "ordered_local_test" if active else "stopped_below_first_failure"
            if active and bool(row["local_r_pass"]):
                z.at[ridx, "fixed_sequence_r_pass"] = True
            else:
                z.at[ridx, "fixed_sequence_r_pass"] = False
                if active:
                    active = False
    return z


def _aggregate_fixed_sequence_regimes(
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
        required = _required_fold_support(total, min_folds, min_fold_fraction)
        support = int(g["fixed_sequence_r_pass"].astype(bool).sum())
        local_support = int(g["local_r_pass"].astype(bool).sum())
        root_free = int((g["train_n_roots"] == 0).sum()) if "train_n_roots" in g.columns else 0
        status = "R" if support >= required else "UNRESOLVED"
        rows.append({
            "return_mode": mode,
            "horizon": int(h),
            "folds_total": total,
            "min_folds_required": required,
            "local_r_folds_support": local_support,
            "fixed_sequence_r_folds_support": support,
            "r_support_fraction": support / float(total) if total else np.nan,
            "root_free_train_folds": root_free,
            "root_free_train_fraction": root_free / float(total) if total else np.nan,
            "certified_regime": status,
            "certified_r_only": bool(status == "R"),
            "certified_m_only": False,
        })
    return pd.DataFrame(rows)


def _next_at_or_after(horizons: np.ndarray, value: int) -> int | None:
    q = horizons[horizons >= int(value)]
    return int(q[0]) if len(q) else None


def _refined_onset(
    fine_regimes: pd.DataFrame,
    coarse_regimes: pd.DataFrame,
    *,
    return_mode: str,
    lower_exclusive: int,
    upper_inclusive: int,
    anchor_spacing_days: int,
    min_consecutive: int,
    min_tail_fraction: float,
) -> dict:
    """Locate the earliest certified point in the coarse-anchored R tail.

    If no interior point survives, the certified coarse upper anchor remains a
    valid one-day grid localization, e.g. (132, 133]. This is a measurement-grid
    bracket, not a confidence interval and not evidence that h<=132 is non-R.
    """
    f = fine_regimes[
        (fine_regimes["return_mode"].astype(str) == str(return_mode))
        & (fine_regimes["horizon"] > int(lower_exclusive))
        & (fine_regimes["horizon"] <= int(upper_inclusive))
    ].copy().sort_values("horizon")
    c = coarse_regimes[coarse_regimes["return_mode"].astype(str) == str(return_mode)].copy()
    c = c[c["horizon"] > int(upper_inclusive)].copy().sort_values("horizon")

    empty = {
        "return_mode": return_mode,
        "identified": False,
        "reason": "no_refined_persistent_r_onset",
        "hc_grid": None,
        "hc_lower_exclusive": None,
        "hc_upper_inclusive": None,
        "hc_midpoint": None,
        "anchor_spacing_days": int(anchor_spacing_days),
        "min_consecutive": int(min_consecutive),
        "tail_r_fraction": None,
        "strict_r_tail": False,
        "later_m_certified": None,
    }
    if f.empty:
        return {**empty, "reason": "no_refinement_rows"}

    # Require the upper anchor to remain certified by the coarse-anchored fold
    # gatekeeping. Then walk downward through the contiguous aggregate R run.
    state = dict(zip(f["horizon"].astype(int), f["certified_regime"].astype(str)))
    if state.get(int(upper_inclusive)) != "R":
        return {**empty, "reason": "coarse_upper_anchor_not_replicated_at_required_fold_support"}

    fine_hs = sorted(int(x) for x in f["horizon"].unique())
    onset = int(upper_inclusive)
    for h in sorted((x for x in fine_hs if x < int(upper_inclusive)), reverse=True):
        if h != onset - 1:
            break
        if state.get(h) != "R":
            break
        onset = h

    combined = pd.concat(
        [f[["horizon", "certified_regime"]], c[["horizon", "certified_regime"]]],
        ignore_index=True,
    ).drop_duplicates(subset=["horizon"], keep="first").sort_values("horizon")
    hs = combined["horizon"].to_numpy(dtype=int)
    state_map = dict(zip(combined["horizon"].astype(int), combined["certified_regime"].astype(str)))

    anchors: list[int] = [onset]
    for j in range(1, max(1, int(min_consecutive))):
        target = int(onset) + j * int(anchor_spacing_days)
        a = _next_at_or_after(hs, target)
        if a is None or state_map.get(a) != "R":
            return {**empty, "reason": "refined_onset_lacks_coarse_scale_persistence_anchors"}
        anchors.append(a)

    tail = combined[combined["horizon"] >= int(onset)]
    states = tail["certified_regime"].astype(str)
    later_m = bool(states.eq("M").any())
    r_fraction = float(states.eq("R").mean()) if len(states) else 0.0
    if later_m or r_fraction + 1e-15 < float(min_tail_fraction):
        return {**empty, "reason": "refined_tail_fails_persistence_fraction"}

    prev_h = onset - 1 if onset > int(lower_exclusive) else int(lower_exclusive)
    return {
        "return_mode": return_mode,
        "identified": True,
        "reason": "",
        "hc_grid": int(onset),
        "hc_lower_exclusive": int(prev_h),
        "hc_upper_inclusive": int(onset),
        "hc_midpoint": 0.5 * (int(prev_h) + int(onset)),
        "anchor_spacing_days": int(anchor_spacing_days),
        "persistence_anchors": anchors,
        "min_consecutive": int(min_consecutive),
        "tail_r_fraction": r_fraction,
        "strict_r_tail": bool(states.eq("R").all()),
        "later_m_certified": False,
    }


def _consensus_refinement(mode_estimates: pd.DataFrame) -> dict:
    if mode_estimates.empty:
        return {"identified": False, "reason": "no_mode_estimates"}
    good = mode_estimates[mode_estimates["identified"].astype(bool)].copy()
    expected = set(mode_estimates["return_mode"].astype(str))
    if set(good["return_mode"].astype(str)) != expected:
        return {"identified": False, "reason": "not_all_return_modes_identified"}
    upper = int(good["hc_upper_inclusive"].max())
    row = good.loc[good["hc_upper_inclusive"].astype(int).idxmax()]
    lower = int(row["hc_lower_exclusive"])
    return {
        "identified": True,
        "definition": "first refined bracket by which every return mode has entered the persistent certified R-only tail",
        "hc_lower_exclusive": lower,
        "hc_upper_inclusive": upper,
        "hc_grid": upper,
        "hc_midpoint": 0.5 * (lower + upper),
        "hc_months_approx": float(upper / 21.0),
        "modes": sorted(expected),
    }


def _coarse_surface_reproduction_audit(
    surface: SplineSurface,
    coarse_train: pd.DataFrame,
    coarse_fold_rows: pd.DataFrame,
    *,
    coarse_horizons: Sequence[int],
    root_qlo: float,
    root_qhi: float,
    root_grid_size: int,
    return_mode: str,
    fold: int,
) -> list[dict]:
    """Verify that the refinement uses the same coarse root gate as v6."""
    lo, hi = np.nanquantile(
        coarse_train["psi_primary"].to_numpy(float), [float(root_qlo), float(root_qhi)]
    )
    psi_grid = np.linspace(lo, hi, int(root_grid_size))
    out: list[dict] = []
    for h in coarse_horizons:
        eval_df = pd.DataFrame({
            "psi_primary": psi_grid,
            "horizon": np.full(len(psi_grid), int(h), dtype=float),
        })
        roots = _roots_from_curve(psi_grid, surface.predict(eval_df))
        q = coarse_fold_rows[
            (coarse_fold_rows["return_mode"].astype(str) == str(return_mode))
            & (coarse_fold_rows["fold"].astype(int) == int(fold))
            & (coarse_fold_rows["horizon"].astype(int) == int(h))
        ]
        expected = int(q.iloc[0]["train_n_roots"]) if (not q.empty and pd.notna(q.iloc[0]["train_n_roots"])) else None
        observed = int(len(roots))
        out.append({
            "return_mode": return_mode,
            "fold": int(fold),
            "horizon": int(h),
            "v6_train_n_roots": expected,
            "v8_reproduced_train_n_roots": observed,
            "root_count_match": bool(expected is None or expected == observed),
        })
    return out


def run_hc_refinement(
    panel: pd.DataFrame,
    cfg,
    coarse_dir: Path,
    out_dir: Path,
    *,
    refinement_horizons: Sequence[int] = DEFAULT_REFINEMENT_HORIZONS,
    fit_horizons: Sequence[int] = DEFAULT_REFINEMENT_FIT_HORIZONS,
    lower_exclusive: int = DEFAULT_REFINEMENT_LOWER_EXCLUSIVE,
    upper_inclusive: int = DEFAULT_REFINEMENT_UPPER_INCLUSIVE,
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coarse_summary, coarse_regimes, coarse_folds = _load_coarse_evidence(Path(coarse_dir))

    fine = tuple(sorted(set(int(x) for x in refinement_horizons)))
    requested_fit = tuple(sorted(set(int(x) for x in fit_horizons)))
    coarse_hs = tuple(sorted(int(x) for x in coarse_summary.get("scan_horizons", [])))
    if not coarse_hs:
        coarse_hs = tuple(int(x) for x in DEFAULT_CRITICAL_HORIZONS)
    if int(upper_inclusive) not in coarse_hs:
        raise ValueError(
            f"upper_inclusive={upper_inclusive} must be a v6 coarse anchor; coarse grid={coarse_hs}"
        )
    if not fine:
        raise ValueError("refinement_horizons is empty")
    if min(fine) <= int(lower_exclusive) or max(fine) > int(upper_inclusive):
        raise ValueError("refinement_horizons must lie in (lower_exclusive, upper_inclusive]")

    # Dense rows are required for OOS measurement, but they are explicitly NOT
    # part of the surface fit. This is the central v8 correction.
    required_panel_hs = tuple(sorted(set(coarse_hs).union(fine)))
    required_cols = ["date", "return_mode", "horizon", "psi_primary", "phase_product"]
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"Refinement panel is missing columns {missing_cols}")
    available = set(int(x) for x in panel["horizon"].dropna().unique())
    missing = sorted(set(required_panel_hs) - available)
    if missing:
        raise ValueError(
            f"Panel is missing refinement horizons {missing}. Rebuild with --horizons "
            + ",".join(map(str, required_panel_hs))
        )
    # Dense horizons remain measurement-only; also make exactly one compact
    # ownership copy instead of full-width copy + filtered copy.
    panel = panel.loc[panel["horizon"].isin(required_panel_hs), required_cols].copy()
    if not pd.api.types.is_datetime64_any_dtype(panel["date"].dtype):
        panel["date"] = pd.to_datetime(panel["date"])

    rcfg = RefinementConfig(
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
        spline_knots=rcfg.spline_knots,
        bootstrap_reps=rcfg.bootstrap_reps,
        max_fit_rows=rcfg.max_fit_rows,
        random_seed=rcfg.random_seed,
        cert_alpha=rcfg.cert_alpha,
        cert_min_effect_sigma=rcfg.cert_min_effect_sigma,
        cert_abs_effect=rcfg.cert_abs_effect,
        cert_min_obs_per_side=rcfg.cert_min_obs_per_band,
        cert_min_dates_per_side=rcfg.cert_min_dates_per_band,
        cert_min_folds=rcfg.cert_min_folds,
        cert_min_fold_fraction=rcfg.cert_min_fold_fraction,
        cert_single_phase_bins=rcfg.cert_single_phase_bins,
    )

    max_h = max(coarse_hs)  # exact v6 purge/bootstrap time scale
    block_len = max(1, int(math.ceil(max_h / int(cfg.anchor_stride))))
    fold_rows: list[dict] = []
    audit_rows: list[dict] = []

    for mode_i, (mode, pm) in enumerate(panel.groupby("return_mode", sort=False)):
        # Exact v6 fold calendar: derive blocks only from the coarse design.
        pm_coarse = pm[pm["horizon"].isin(coarse_hs)].copy()
        blocks = _walkforward_date_blocks(pm_coarse, cfg)
        for fold, (test_start, test_end) in enumerate(blocks, start=1):
            purge = pd.offsets.BDay(max_h)
            train = pm[pm["date"] < (test_start - purge)].copy()
            test = pm[(pm["date"] >= test_start) & (pm["date"] <= test_end)].copy()
            coarse_train = _coarse_fit_subset(train, coarse_hs)
            if len(coarse_train) < 1000 or len(test) < 200:
                continue

            # Exact v6 seed + exact v6 coarse-horizon sampling distribution.
            fit_seed = _v6_fit_seed(rcfg.random_seed, mode_i, fold)
            fit_train = _balanced_fit_sample(coarse_train, rcfg.max_fit_rows, fit_seed)
            surface = SplineSurface(
                task="continuous",
                interaction=True,
                context=False,
                n_knots=rcfg.spline_knots,
                degree=3,
                ridge_alpha=4.0,
                logistic_c=0.5,
            ).fit(fit_train, _date_weights(fit_train))

            audit_rows.extend(_coarse_surface_reproduction_audit(
                surface,
                coarse_train,
                coarse_folds,
                coarse_horizons=coarse_hs,
                root_qlo=rcfg.root_domain_qlo,
                root_qhi=rcfg.root_domain_qhi,
                root_grid_size=rcfg.root_grid_size,
                return_mode=str(mode),
                fold=fold,
            ))

            lo, hi = np.nanquantile(
                coarse_train["psi_primary"].to_numpy(float),
                [rcfg.root_domain_qlo, rcfg.root_domain_qhi],
            )
            psi_grid = np.linspace(lo, hi, rcfg.root_grid_size)

            for h in fine:
                train_h = train[train["horizon"] == int(h)]
                test_h = test[test["horizon"] == int(h)]
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
                    "horizon": np.full(len(psi_grid), int(h), dtype=float),
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
                    "surface_fit_design": "frozen_v6_coarse_horizons",
                    **single,
                })
            print(f"hc-refinement-v8 {mode} fold {fold}/{len(blocks)} complete")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "hc_refinement_coarse_reproduction_audit.csv", index=False)
    if not audit.empty and not bool(audit["root_count_match"].astype(bool).all()):
        bad = audit[~audit["root_count_match"].astype(bool)]
        raise RuntimeError(
            "v8 coarse-fit invariance audit failed: refinement does not reproduce v6 root counts. "
            f"See {out_dir / 'hc_refinement_coarse_reproduction_audit.csv'}; mismatches={len(bad)}"
        )

    folds = pd.DataFrame(fold_rows)
    folds = _fixed_sequence_gatekeeping(
        folds,
        coarse_folds,
        upper_anchor=int(upper_inclusive),
        alpha=rcfg.cert_alpha,
    )
    regimes = _aggregate_fixed_sequence_regimes(
        folds,
        min_folds=rcfg.cert_min_folds,
        min_fold_fraction=rcfg.cert_min_fold_fraction,
    )

    local_steps = [b - a for a, b in zip(coarse_hs[:-1], coarse_hs[1:]) if a >= 112 and b <= 140]
    anchor_spacing = int(round(float(np.median(local_steps)))) if local_steps else 7

    mode_rows: list[dict] = []
    tail_guards: list[dict] = []
    modes = sorted(set(coarse_regimes["return_mode"].astype(str)))
    for mode in modes:
        guard = _coarse_tail_guard(coarse_regimes, return_mode=mode, upper=upper_inclusive)
        tail_guards.append({"return_mode": mode, **guard})
        if not guard["ok"]:
            mode_rows.append({"return_mode": mode, "identified": False, "reason": guard["reason"]})
            continue
        mode_rows.append(_refined_onset(
            regimes,
            coarse_regimes,
            return_mode=mode,
            lower_exclusive=lower_exclusive,
            upper_inclusive=upper_inclusive,
            anchor_spacing_days=anchor_spacing,
            min_consecutive=rcfg.hc_min_consecutive,
            min_tail_fraction=rcfg.hc_min_tail_fraction,
        ))

    mode_estimates = pd.DataFrame(mode_rows)
    consensus = _consensus_refinement(mode_estimates)
    tail_guard_df = pd.DataFrame(tail_guards)

    folds.to_csv(out_dir / "hc_refinement_fold_certification.csv", index=False)
    regimes.to_csv(out_dir / "hc_refinement_regimes.csv", index=False)
    mode_estimates.to_csv(out_dir / "hc_refinement_estimates.csv", index=False)
    tail_guard_df.to_csv(out_dir / "hc_refinement_coarse_tail_guard.csv", index=False)

    summary = {
        "status": "EXPLORATORY_RESOLUTION_REFINEMENT_ONLY_V8_METHOD_CORRECTION",
        "definition": "refine h_c inside a pre-identified bracket while freezing the v6 coarse surface fit and using ordered fixed-sequence gatekeeping from the certified upper anchor",
        "method_correction": {
            "problem_in_v7": "dense horizons were allowed to alter the spline fit/root-free gate and were re-penalized as an unordered Holm family",
            "surface_fit": "exact v6 coarse horizons only; exact v6 seed scheme; dense horizons are measurement-only",
            "multiplicity": "fixed-sequence gatekeeping from the v6-certified upper anchor within each return-mode x fold",
            "local_certification_changed": False,
            "alpha_changed": False,
            "effect_floor_changed": False,
            "bootstrap_changed": False,
            "fold_support_changed": False,
        },
        "refinement_interval": {
            "lower_exclusive": int(lower_exclusive),
            "upper_inclusive": int(upper_inclusive),
        },
        "refinement_horizons": list(fine),
        "measurement_panel_horizons": list(required_panel_hs),
        "fit_horizons": list(required_panel_hs),  # compatibility: horizons required in the panel
        "surface_fit_horizons": list(coarse_hs),
        "ignored_dense_fit_horizons_from_cli_for_backward_compatibility": [
            int(x) for x in requested_fit if int(x) not in set(coarse_hs)
        ],
        "discovery_coarse_scan_horizons": list(coarse_hs),
        "persistence_anchor_spacing_days": anchor_spacing,
        "multiplicity": "ordered fixed-sequence gatekeeping from coarse-certified upper anchor; local IUT alpha unchanged",
        "coarse_fit_invariance_audit": {
            "rows": int(len(audit)),
            "all_root_counts_match_v6": bool(audit.empty or audit["root_count_match"].astype(bool).all()),
            "file": "hc_refinement_coarse_reproduction_audit.csv",
        },
        "certification_rules_unchanged": {
            "spline_knots": rcfg.spline_knots,
            "bootstrap_reps": rcfg.bootstrap_reps,
            "max_fit_rows": rcfg.max_fit_rows,
            "alpha": rcfg.cert_alpha,
            "min_effect_sigma": rcfg.cert_min_effect_sigma,
            "abs_effect": rcfg.cert_abs_effect,
            "min_obs_per_band": rcfg.cert_min_obs_per_band,
            "min_dates_per_band": rcfg.cert_min_dates_per_band,
            "min_folds": rcfg.cert_min_folds,
            "min_fold_fraction": rcfg.cert_min_fold_fraction,
            "single_phase_bins": rcfg.cert_single_phase_bins,
            "min_consecutive": rcfg.hc_min_consecutive,
            "min_tail_fraction": rcfg.hc_min_tail_fraction,
            "walkforward_splits": int(cfg.n_walkforward_splits),
            "min_train_frac": float(cfg.min_train_frac),
            "stride": int(cfg.anchor_stride),
            "seed": int(cfg.random_seed),
        },
        "per_mode_hc_refined": mode_estimates.to_dict("records"),
        "consensus_hc_refined": consensus,
        "guardrail": "This v8 re-analysis corrects a pre-validation methodological bug in v7. Results on the discovery sample remain exploratory; re-freeze the independent-validation protocol before touching validation data.",
    }
    (out_dir / "hc_refinement_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary

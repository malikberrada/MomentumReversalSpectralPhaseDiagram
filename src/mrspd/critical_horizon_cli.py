from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .critical_horizon import DEFAULT_CRITICAL_HORIZONS, run_critical_horizon_scan


def _parse_horizons(text: str) -> tuple[int, ...]:
    vals = tuple(sorted(set(int(x.strip()) for x in text.split(",") if x.strip())))
    if len(vals) < 3 or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("provide at least three positive comma-separated horizons")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate the MRSPD critical horizon h_c from a dense-horizon panel"
    )
    p.add_argument("--panel", required=True)
    p.add_argument("--out", default=None, help="defaults to PANEL_DIR/critical-horizon")
    p.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=DEFAULT_CRITICAL_HORIZONS,
        help="comma-separated scan grid",
    )
    p.add_argument("--phase-knots", type=int, default=7)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--phase-max-fit-rows", type=int, default=250000)
    p.add_argument("--cert-alpha", type=float, default=0.05)
    p.add_argument("--cert-min-effect-sigma", type=float, default=0.02)
    p.add_argument("--cert-abs-effect", type=float, default=0.0)
    p.add_argument("--cert-min-obs-per-band", type=int, default=250)
    p.add_argument("--cert-min-dates-per-band", type=int, default=40)
    p.add_argument("--cert-min-folds", type=int, default=3)
    p.add_argument("--cert-min-fold-fraction", type=float, default=0.75)
    p.add_argument("--cert-single-phase-bins", type=int, default=5)
    p.add_argument("--hc-min-consecutive", type=int, default=3)
    p.add_argument("--hc-min-tail-fraction", type=float, default=0.80)
    p.add_argument("--walkforward-splits", type=int, default=4)
    p.add_argument("--min-train-frac", type=float, default=0.50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260812)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    out_dir = Path(args.out) if args.out else panel_path.parent / "critical-horizon"
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    cfg = SimpleNamespace(
        horizons=tuple(args.horizons),
        spectral_spans=tuple(args.horizons),
        min_train_frac=args.min_train_frac,
        n_walkforward_splits=args.walkforward_splits,
        anchor_stride=args.stride,
        random_seed=args.seed,
    )
    summary = run_critical_horizon_scan(
        panel,
        cfg,
        out_dir,
        horizons=args.horizons,
        spline_knots=args.phase_knots,
        bootstrap_reps=args.bootstrap_reps,
        max_fit_rows=args.phase_max_fit_rows,
        cert_alpha=args.cert_alpha,
        cert_min_effect_sigma=args.cert_min_effect_sigma,
        cert_abs_effect=args.cert_abs_effect,
        cert_min_obs_per_band=args.cert_min_obs_per_band,
        cert_min_dates_per_band=args.cert_min_dates_per_band,
        cert_min_folds=args.cert_min_folds,
        cert_min_fold_fraction=args.cert_min_fold_fraction,
        cert_single_phase_bins=args.cert_single_phase_bins,
        hc_min_consecutive=args.hc_min_consecutive,
        hc_min_tail_fraction=args.hc_min_tail_fraction,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

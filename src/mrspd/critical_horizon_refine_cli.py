from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .critical_horizon_refine import (
    DEFAULT_REFINEMENT_FIT_HORIZONS,
    DEFAULT_REFINEMENT_HORIZONS,
    DEFAULT_REFINEMENT_LOWER_EXCLUSIVE,
    DEFAULT_REFINEMENT_UPPER_INCLUSIVE,
    run_hc_refinement,
)


def _parse_horizons(text: str) -> tuple[int, ...]:
    vals = tuple(sorted(set(int(x.strip()) for x in text.split(",") if x.strip())))
    if not vals or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("provide positive comma-separated horizons")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine MRSPD h_c inside a pre-identified bracket")
    p.add_argument("--panel", required=True)
    p.add_argument("--coarse-dir", required=True, help="v6 critical-horizon output directory")
    p.add_argument("--out", required=True)
    p.add_argument("--refinement-horizons", type=_parse_horizons,
                   default=DEFAULT_REFINEMENT_HORIZONS)
    p.add_argument("--fit-horizons", type=_parse_horizons,
                   default=DEFAULT_REFINEMENT_FIT_HORIZONS)
    p.add_argument("--lower-exclusive", type=int, default=DEFAULT_REFINEMENT_LOWER_EXCLUSIVE)
    p.add_argument("--upper-inclusive", type=int, default=DEFAULT_REFINEMENT_UPPER_INCLUSIVE)
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
    panel = pd.read_csv(Path(args.panel), parse_dates=["date"])
    cfg = SimpleNamespace(
        horizons=tuple(args.fit_horizons),
        spectral_spans=tuple(args.fit_horizons),
        min_train_frac=args.min_train_frac,
        n_walkforward_splits=args.walkforward_splits,
        anchor_stride=args.stride,
        random_seed=args.seed,
    )
    summary = run_hc_refinement(
        panel,
        cfg,
        Path(args.coarse_dir),
        Path(args.out),
        refinement_horizons=args.refinement_horizons,
        fit_horizons=args.fit_horizons,
        lower_exclusive=args.lower_exclusive,
        upper_inclusive=args.upper_inclusive,
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

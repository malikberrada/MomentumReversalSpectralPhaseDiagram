from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .phase_surface import run_phase_surface_analysis


def _infer_spans(panel: pd.DataFrame) -> tuple[int, ...]:
    spans = []
    for c in panel.columns:
        if c.startswith("psi_") and c != "psi_primary":
            try:
                spans.append(int(c.split("_", 1)[1]))
            except ValueError:
                pass
    return tuple(sorted(set(spans)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run only the non-monotone MRSPD phase-surface analysis from an existing panel.csv.gz"
    )
    p.add_argument("--panel", required=True)
    p.add_argument("--out", default=None, help="defaults to the panel directory")
    p.add_argument("--phase-bins", type=int, default=30)
    p.add_argument("--phase-knots", type=int, default=7)
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--phase-max-fit-rows", type=int, default=250000)
    p.add_argument("--cert-alpha", type=float, default=0.05)
    p.add_argument("--cert-side-width-q", type=float, default=0.08)
    p.add_argument("--cert-side-gap-q", type=float, default=0.01)
    p.add_argument("--cert-min-effect-sigma", type=float, default=0.02)
    p.add_argument("--cert-abs-effect", type=float, default=0.0)
    p.add_argument("--cert-min-obs-per-side", type=int, default=250)
    p.add_argument("--cert-min-dates-per-side", type=int, default=40)
    p.add_argument("--cert-min-folds", type=int, default=3)
    p.add_argument("--cert-min-fold-fraction", type=float, default=0.75)
    p.add_argument("--cert-root-cluster-q", type=float, default=0.06)
    p.add_argument("--cert-max-root-iqr-q", type=float, default=0.05)
    p.add_argument("--cert-single-phase-bins", type=int, default=5)
    p.add_argument("--walkforward-splits", type=int, default=4)
    p.add_argument("--min-train-frac", type=float, default=0.50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260812)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    out_dir = Path(args.out) if args.out else panel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    horizons = tuple(sorted(int(x) for x in panel["horizon"].dropna().unique()))
    spans = _infer_spans(panel)
    if not horizons:
        raise RuntimeError("No horizons found in panel")
    if not spans:
        spans = horizons

    cfg = SimpleNamespace(
        horizons=horizons,
        spectral_spans=spans,
        min_train_frac=args.min_train_frac,
        n_walkforward_splits=args.walkforward_splits,
        anchor_stride=args.stride,
        random_seed=args.seed,
    )
    summary = run_phase_surface_analysis(
        panel,
        cfg,
        out_dir,
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

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            full = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            full = {}
        full["nonmonotone_phase_surface"] = summary
        summary_path.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

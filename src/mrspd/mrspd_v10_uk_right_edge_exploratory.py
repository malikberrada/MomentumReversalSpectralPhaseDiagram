#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .robust_transport_v9_2 import (
    HUBER_C,
    MAD_NORMALIZER,
    load_panel_streaming,
    add_global_percentile_and_robust_response,
    certify_candidate,
    aggregate_regimes,
    identify_hc,
    consensus_for_q0,
)

FROZEN_Q0 = 0.85
DEFAULT_EXTENDED_HORIZONS = [
    63, 72, 84, 96, 105, 112, 119, 126,
    133, 140, 147, 168, 189, 210, 231, 252,
]


def parse_horizons(s: str) -> list[int]:
    xs = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(xs) != len(set(xs)):
        raise ValueError("duplicate horizons")
    if xs != sorted(xs):
        raise ValueError("horizons must be increasing")
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MRSPD v10 development-only right-edge extension. "
            "Tests the already-selected robust upper tail q>=0.85 on the "
            "consumed UK V3 universe. This is NOT a confirmatory rerun."
        )
    )
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--horizons",
        default=",".join(map(str, DEFAULT_EXTENDED_HORIZONS)),
    )
    args = ap.parse_args()

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    hyp = protocol["hypothesis"]
    rules = protocol["rules"]

    if abs(float(hyp["q0"]) - FROZEN_Q0) > 1e-12:
        raise ValueError(
            f"expected previously selected q0={FROZEN_Q0}, got {hyp['q0']}"
        )
    if abs(float(hyp["huber_c"]) - HUBER_C) > 1e-12:
        raise ValueError("Huber c mismatch")
    if abs(float(hyp["mad_normalizer"]) - MAD_NORMALIZER) > 1e-12:
        raise ValueError("MAD normalizer mismatch")

    horizons = parse_horizons(args.horizons)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_panel_streaming(Path(args.panel), horizons=horizons)
    pre = add_global_percentile_and_robust_response(raw, huber_c=HUBER_C)
    del raw

    folds = certify_candidate(
        pre,
        universe="uk_v3_consumed_exploratory",
        q0=FROZEN_Q0,
        horizons=horizons,
        splits=int(rules["walkforward_splits"]),
        min_train_frac=float(rules["min_train_frac"]),
        bootstrap_reps=int(rules["bootstrap_reps"]),
        alpha=float(rules["alpha"]),
        min_effect_sigma=float(rules["min_effect_sigma"]),
        min_obs_per_bin=250,
        min_dates_per_bin=40,
        min_train_dates_per_bin=40,
        seed=int(rules["seed"]),
    )
    regs = aggregate_regimes(
        folds,
        min_folds=int(rules["min_folds"]),
        min_fold_fraction=float(rules["min_fold_fraction"]),
    )
    hc_rows = identify_hc(
        regs,
        horizons,
        min_consecutive=int(rules["hc_min_consecutive"]),
        min_tail_fraction=float(rules["hc_min_tail_fraction"]),
    )
    consensus = consensus_for_q0(
        hc_rows,
        "uk_v3_consumed_exploratory",
        FROZEN_Q0,
    )

    folds.to_csv(out / "v10_uk_fold_certification.csv", index=False)
    regs.to_csv(out / "v10_uk_regimes.csv", index=False)
    pd.DataFrame(hc_rows).to_csv(out / "v10_uk_hc.csv", index=False)

    mode_hc = {}
    for row in hc_rows:
        if row.get("universe") != "uk_v3_consumed_exploratory":
            continue
        if abs(float(row.get("q0", FROZEN_Q0)) - FROZEN_Q0) > 1e-12:
            continue
        mode = str(row.get("return_mode"))
        mode_hc[mode] = (
            int(row["hc_grid"])
            if row.get("identified") and row.get("hc_grid") is not None
            else None
        )

    right = regs[regs["horizon"].isin([147, 168, 189, 210, 231, 252])].copy()
    right.to_csv(out / "v10_uk_right_edge_regimes.csv", index=False)

    result = {
        "status": "V10_DEVELOPMENT_ONLY_RIGHT_EDGE_EXTENSION",
        "confirmatory_v9_2_verdict_unchanged": True,
        "frozen_from_v9_2": {
            "q0": FROZEN_Q0,
            "huber_c": HUBER_C,
            "mad_normalizer": MAD_NORMALIZER,
            "alpha": rules["alpha"],
            "bootstrap_reps": rules["bootstrap_reps"],
            "min_effect_sigma": rules["min_effect_sigma"],
            "min_folds": rules["min_folds"],
            "min_fold_fraction": rules["min_fold_fraction"],
            "hc_min_consecutive": rules["hc_min_consecutive"],
            "hc_min_tail_fraction": rules["hc_min_tail_fraction"],
        },
        "extended_horizons": horizons,
        "mode_hc": mode_hc,
        "consensus": consensus,
        "decision": (
            "CANDIDATE_FOR_V10_CROSS_MARKET_HETEROGENEOUS_HC"
            if all(mode_hc.get(m) is not None for m in ["raw", "market_residual"])
            else "ROBUST_TAIL_FAMILY_NOT_YET_PERSISTENT_IN_BOTH_UK_MODES"
        ),
        "guardrail": (
            "This run is exploratory because UK V3 has already been observed. "
            "It cannot change the v9.2 NOT_CONFIRMED verdict. "
            "Any v10 hypothesis selected from D+V2+V3 must be frozen before an untouched V4."
        ),
    }
    (out / "v10_uk_right_edge_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

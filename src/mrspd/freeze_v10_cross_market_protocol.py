#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

FROZEN_Q0 = 0.85
HUBER_C = 1.345
MAD_NORMALIZER = 1.4826

def canonical_sha(doc: dict) -> str:
    x = dict(doc)
    x.pop("protocol_sha256", None)
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous-protocol", required=True)
    ap.add_argument("--uk-summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prev = json.loads(Path(a.previous_protocol).read_text(encoding="utf-8"))
    uk = json.loads(Path(a.uk_summary).read_text(encoding="utf-8"))

    if uk.get("decision") != "CANDIDATE_FOR_V10_CROSS_MARKET_HETEROGENEOUS_HC":
        raise SystemExit("V10_FREEZE: FAIL UK exploratory result is not an identified v10 candidate")
    cons = uk.get("consensus") or {}
    mode_hc = uk.get("mode_hc") or {}
    if not cons.get("identified") or mode_hc.get("raw") is None or mode_hc.get("market_residual") is None:
        raise SystemExit("V10_FREEZE: FAIL both UK modes must have identified onsets")

    frozen = uk.get("frozen_from_v9_2") or {}
    if abs(float(frozen.get("q0", FROZEN_Q0)) - FROZEN_Q0) > 1e-12:
        raise SystemExit("V10_FREEZE: FAIL q0 mismatch")
    if abs(float(frozen.get("huber_c", HUBER_C)) - HUBER_C) > 1e-12:
        raise SystemExit("V10_FREEZE: FAIL Huber constant mismatch")
    if abs(float(frozen.get("mad_normalizer", MAD_NORMALIZER)) - MAD_NORMALIZER) > 1e-12:
        raise SystemExit("V10_FREEZE: FAIL MAD normalizer mismatch")

    horizons = list(map(int, uk["extended_horizons"]))
    if horizons != sorted(set(horizons)) or len(horizons) < 3:
        raise SystemExit("V10_FREEZE: FAIL invalid horizon grid")

    rules = {
        "bootstrap_reps": int(frozen["bootstrap_reps"]),
        "alpha": float(frozen["alpha"]),
        "min_effect_sigma": float(frozen["min_effect_sigma"]),
        "effect_scale": "train SD of daily robust tail-bin means",
        "tail_bins": 2,
        "min_obs_per_tail_bin": 250,
        "min_dates_per_tail_bin": 40,
        "min_train_dates_per_tail_bin": 40,
        "min_folds": int(frozen["min_folds"]),
        "min_fold_fraction": float(frozen["min_fold_fraction"]),
        "hc_min_consecutive": int(frozen["hc_min_consecutive"]),
        "hc_min_tail_fraction": float(frozen["hc_min_tail_fraction"]),
        "walkforward_splits": int(prev["rules"]["walkforward_splits"]),
        "min_train_frac": float(prev["rules"]["min_train_frac"]),
        "seed": int(prev["rules"]["seed"]),
        "huber_c": HUBER_C,
        "mad_normalizer": MAD_NORMALIZER,
    }

    doc = {
        "status": "FROZEN_V10_BEFORE_V4_PANEL_CONSTRUCTION",
        "version": "MRSPD-CROSS-MARKET-ROBUST-TAIL-v10",
        "parent_v9_2_protocol_sha256": prev.get("protocol_sha256"),
        "hypothesis": {
            "statement": (
                "In a previously unseen national equity universe, the upper global spectral-percentile "
                "tail q_psi>=0.85 exhibits a persistent negative robust phase-product regime in both raw "
                "and local-market-residual modes; the critical onset h_c is market-specific rather than universal."
            ),
            "normalization": "global empirical percentile within universe x date x return_mode x horizon",
            "tail_direction": "upper",
            "q0": FROZEN_Q0,
            "tail_bins": 2,
            "tail_bin_edges": [0.85, 0.925, 1.0],
            "response": "cross_sectional_huber_clipped_phase_product",
            "huber_c": HUBER_C,
            "mad_normalizer": MAD_NORMALIZER,
            "market_residual_rule": (
                "raw_log_return - rolling_beta(raw_log_return, local_market_log_return) * local_market_log_return"
            ),
            "market_benchmark_principle": "freeze one broad domestic equity-market benchmark before V4 prices",
        },
        "horizons": horizons,
        "rules": rules,
        "primary_confirmatory_endpoint": {
            "phenomenon": (
                "persistent certified negative upper-tail regime must have an identified h_c in BOTH "
                "raw and local-market-residual modes"
            ),
            "localization_gate": None,
            "reason_no_localization_gate": (
                "v10 explicitly hypothesizes market-specific h_c; h_c is estimated out of sample on the "
                "pre-frozen finite horizon grid rather than forced to the US or UK development location"
            ),
            "overall_pass": (
                "design_audit_pass AND raw_hc_identified AND market_residual_hc_identified"
            ),
        },
        "development_evidence": {
            "uk_consumed_exploratory_mode_hc": {
                "raw": int(mode_hc["raw"]),
                "market_residual": int(mode_hc["market_residual"]),
            },
            "uk_consumed_exploratory_consensus_hc": int(cons["hc_grid"]),
            "v9_2_confirmatory_verdict_remains": "NOT_CONFIRMED",
            "interpretation": (
                "development evidence motivates cross-market transport with heterogeneous h_c; "
                "it is not itself confirmatory evidence for v10"
            ),
        },
        "v4_guardrails": {
            "candidate_ticker_list_must_be_bound_before_price_panel": True,
            "minimum_panel_coverage_fraction": 0.80,
            "no_replacements_after_binding": True,
            "zero_ticker_overlap_with_development_universes": True,
            "local_market_benchmark_must_be_bound_before_prices": True,
            "do_not_change_q0_huber_horizons_alpha_holm_fold_or_persistence_rules_after_v4": True,
        },
    }
    doc["protocol_sha256"] = canonical_sha(doc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": doc["status"],
        "version": doc["version"],
        "q0": FROZEN_Q0,
        "horizons": horizons,
        "localization_gate": None,
        "protocol_sha256": doc["protocol_sha256"],
        "out": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()

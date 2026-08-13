#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def parse_bins(x):
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(
        description="Forensic, outcome-preserving audit of MRSPD V3 v9.2 confirmatory failure."
    )
    ap.add_argument("--confirmatory-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cdir = Path(a.confirmatory_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fold_path = cdir / "third_robust_tail_fold_certification.csv"
    regime_path = cdir / "third_robust_tail_regimes.csv"
    hc_path = cdir / "third_robust_tail_hc.csv"
    summary_path = cdir / "third_universe_validation_summary_v9_2.json"
    audit_path = cdir / "third_universe_v9_2_audit.json"

    for p in [fold_path, regime_path, hc_path, summary_path, audit_path]:
        if not p.exists():
            raise SystemExit(f"MISSING: {p}")

    protocol = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    verdict = json.loads(summary_path.read_text(encoding="utf-8"))
    design_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    f = pd.read_csv(fold_path)
    r = pd.read_csv(regime_path)
    h = pd.read_csv(hc_path)

    q0 = float(protocol["hypothesis"]["q0"])
    horizons = list(map(int, protocol["horizons"]))
    min_folds = int(protocol["rules"]["min_folds"])
    min_frac = float(protocol["rules"]["min_fold_fraction"])
    min_consec = int(protocol["rules"]["hc_min_consecutive"])
    min_tail_frac = float(protocol["rules"]["hc_min_tail_fraction"])

    f = f[np.isclose(pd.to_numeric(f["q0"], errors="coerce"), q0)].copy()
    if f.empty:
        raise SystemExit(f"No fold rows for frozen q0={q0}")

    bool_cols = [
        "eligible", "tail_negative_local", "passes_holm",
        "certified_tail_negative_fold"
    ]
    for c in bool_cols:
        f[c] = f[c].map(as_bool)

    # Invariant audit at fold level.
    inv_rows = []
    detail_rows = []
    for _, row in f.iterrows():
        bins = parse_bins(row.get("tail_bins_json", "[]"))
        bin_local = []
        for b in bins:
            eff = as_bool(b.get("effect_floor_pass"))
            ci = as_bool(b.get("ci_negative_pass"))
            est = pd.to_numeric(pd.Series([b.get("estimate")]), errors="coerce").iloc[0]
            ci_hi = pd.to_numeric(pd.Series([b.get("ci_hi")]), errors="coerce").iloc[0]
            p = pd.to_numeric(pd.Series([b.get("p_lt0")]), errors="coerce").iloc[0]
            epsilon = pd.to_numeric(pd.Series([b.get("epsilon")]), errors="coerce").iloc[0]
            bin_local.append(eff and ci)
            detail_rows.append({
                "return_mode": row["return_mode"],
                "fold": int(row["fold"]),
                "horizon": int(row["horizon"]),
                "bin": int(b.get("bin", -1)),
                "q_lo": b.get("q_lo"),
                "q_hi": b.get("q_hi"),
                "estimate": est,
                "epsilon": epsilon,
                "ci_hi": ci_hi,
                "p_lt0": p,
                "effect_floor_pass": eff,
                "ci_negative_pass": ci,
                "estimate_negative": bool(np.isfinite(est) and est < 0),
            })

        expected_local = bool(row["eligible"] and bins and all(bin_local))
        expected_cert = bool(expected_local and row["passes_holm"])
        inv_rows.append({
            "return_mode": row["return_mode"],
            "fold": int(row["fold"]),
            "horizon": int(row["horizon"]),
            "eligible": row["eligible"],
            "stored_local": row["tail_negative_local"],
            "expected_local": expected_local,
            "local_match": bool(row["tail_negative_local"] == expected_local),
            "passes_holm": row["passes_holm"],
            "stored_certified": row["certified_tail_negative_fold"],
            "expected_certified": expected_cert,
            "certified_match": bool(row["certified_tail_negative_fold"] == expected_cert),
        })

    inv = pd.DataFrame(inv_rows)
    bins = pd.DataFrame(detail_rows)
    inv.to_csv(out / "fold_logic_invariants.csv", index=False)
    bins.to_csv(out / "bin_failure_details.csv", index=False)

    # Summarize precise failure mechanism by mode/horizon.
    rows = []
    for (mode, horizon), g in f.groupby(["return_mode", "horizon"], observed=True):
        bd = bins[(bins["return_mode"] == mode) & (bins["horizon"] == int(horizon))]
        fold_local = int(g["tail_negative_local"].sum())
        fold_holm = int(g["passes_holm"].sum())
        fold_cert = int(g["certified_tail_negative_fold"].sum())

        # Count individual bins failing each gate.
        neg_fail = int((~bd["estimate_negative"]).sum()) if len(bd) else 0
        effect_fail = int((~bd["effect_floor_pass"]).sum()) if len(bd) else 0
        ci_fail = int((~bd["ci_negative_pass"]).sum()) if len(bd) else 0

        eligible = int(g["eligible"].sum())
        frac = fold_cert / max(1, eligible)
        regime_expected = bool(fold_cert >= min_folds and frac >= min_frac)

        if eligible < min_folds:
            mechanism = "ELIGIBILITY"
        elif fold_local < min_folds:
            if neg_fail:
                mechanism = "LOCAL_SIGN_OR_MAGNITUDE"
            elif effect_fail:
                mechanism = "EFFECT_FLOOR"
            elif ci_fail:
                mechanism = "CI"
            else:
                mechanism = "LOCAL_IUT"
        elif fold_cert < min_folds:
            mechanism = "HOLM_ONLY"
        else:
            mechanism = "CERTIFIED"

        rows.append({
            "return_mode": mode,
            "horizon": int(horizon),
            "eligible_folds": eligible,
            "local_negative_folds": fold_local,
            "holm_pass_folds": fold_holm,
            "certified_folds": fold_cert,
            "support_fraction": frac,
            "expected_regime_certified": regime_expected,
            "bin_nonnegative_count": neg_fail,
            "bin_effect_floor_fail_count": effect_fail,
            "bin_ci_fail_count": ci_fail,
            "dominant_failure_mechanism": mechanism,
            "median_candidate_p_iut": pd.to_numeric(g["candidate_p_iut"], errors="coerce").median(),
        })

    diag = pd.DataFrame(rows).sort_values(["return_mode", "horizon"])
    diag.to_csv(out / "horizon_failure_mechanisms.csv", index=False)

    # Cross-check stored regime table.
    rr = r[np.isclose(pd.to_numeric(r["q0"], errors="coerce"), q0)].copy()
    rr["certified_tail_negative"] = rr["certified_tail_negative"].map(as_bool)
    merged = diag.merge(
        rr[["return_mode", "horizon", "certified_folds",
            "support_fraction", "certified_tail_negative"]],
        on=["return_mode", "horizon"],
        how="left",
        suffixes=("_recomputed", "_stored"),
    )
    merged["regime_certified_match"] = (
        merged["expected_regime_certified"] ==
        merged["certified_tail_negative"].fillna(False)
    )
    merged["certified_folds_match"] = (
        pd.to_numeric(merged["certified_folds_recomputed"], errors="coerce") ==
        pd.to_numeric(merged["certified_folds_stored"], errors="coerce")
    )
    merged.to_csv(out / "regime_logic_invariants.csv", index=False)

    # Recompute onset per mode from stored certified regimes.
    onset = {}
    for mode in ["market_residual", "raw"]:
        gm = rr[rr["return_mode"].astype(str) == mode]
        mp = {
            int(x.horizon): as_bool(x.certified_tail_negative)
            for x in gm.itertuples()
        }
        found = None
        for i, hh in enumerate(horizons):
            tail = horizons[i:]
            if len(tail) < min_consec:
                continue
            if not all(mp.get(x, False) for x in tail[:min_consec]):
                continue
            frac = sum(mp.get(x, False) for x in tail) / len(tail)
            if frac >= min_tail_frac:
                found = int(hh)
                break
        onset[mode] = found

    consensus = (
        max(onset.values())
        if all(onset[m] is not None for m in ["market_residual", "raw"])
        else None
    )

    # Diagnostic only: nearest-to-pass horizons and right-edge censoring.
    nearest = {}
    for mode in ["market_residual", "raw"]:
        x = diag[diag["return_mode"] == mode].copy()
        if len(x):
            x = x.sort_values(
                ["certified_folds", "local_negative_folds", "horizon"],
                ascending=[False, False, False],
            )
            nearest[mode] = x.head(5).to_dict(orient="records")
        else:
            nearest[mode] = []

    right_edge = {}
    last_h = max(horizons)
    for mode in ["market_residual", "raw"]:
        z = diag[(diag["return_mode"] == mode) & (diag["horizon"] == last_h)]
        right_edge[mode] = z.to_dict(orient="records")[0] if len(z) else None

    all_fold_logic = bool(inv["local_match"].all() and inv["certified_match"].all())
    all_regime_logic = bool(
        merged["regime_certified_match"].all() and
        merged["certified_folds_match"].all()
    )

    statistical_class = "STATISTICAL_NONREPLICATION_UNDER_FROZEN_PROTOCOL"
    if not all_fold_logic or not all_regime_logic:
        statistical_class = "IMPLEMENTATION_INVARIANT_FAILURE"
    elif consensus is not None:
        statistical_class = "PHENOMENON_PRESENT_CHECK_LOCALIZATION"
    else:
        # Still a genuine frozen-protocol nonreplication, but record whether the
        # current horizon grid may be right-censored for a future exploratory study.
        edge_support = [
            (right_edge[m] or {}).get("certified_folds", 0)
            for m in ["market_residual", "raw"]
        ]
        if max(edge_support, default=0) >= min_folds:
            statistical_class = "NONREPLICATION_WITH_RIGHT_EDGE_SIGNAL_FOR_FUTURE_EXPLORATION"

    report = {
        "status": "MRSPD_V9_2_V3_FORENSIC_AUDIT",
        "frozen_q0": q0,
        "overall_confirmatory_pass_original": bool(verdict.get("overall_confirmatory_pass")),
        "original_interpretation": verdict.get("interpretation"),
        "design_audit_pass": all([
            bool(design_audit.get("market_benchmark_pass")),
            bool(design_audit.get("independence_pass")),
            bool(design_audit.get("bound_subset_pass")),
            bool(design_audit.get("coverage_pass")),
            bool(design_audit.get("horizon_design_pass")),
            bool(design_audit.get("return_modes_present_pass")),
        ]),
        "fold_logic_invariants_pass": all_fold_logic,
        "regime_logic_invariants_pass": all_regime_logic,
        "recomputed_mode_hc": onset,
        "recomputed_consensus_hc": consensus,
        "classification": statistical_class,
        "right_edge_horizon": last_h,
        "right_edge_diagnostics": right_edge,
        "nearest_to_pass_by_mode": nearest,
        "guardrail": (
            "This audit does not change q0, Huber constants, horizons, alpha, "
            "bootstrap, Holm, fold support, persistence, or localization. "
            "A frozen V3 NOT_CONFIRMED verdict may only be changed if an "
            "implementation invariant demonstrably fails."
        ),
    }
    (out / "forensic_summary.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

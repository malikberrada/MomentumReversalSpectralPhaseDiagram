from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def canonical_sha256(doc: dict[str, Any]) -> str:
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def required_support(total: int, min_folds: int, min_frac: float) -> int:
    return max(int(min_folds), int(math.ceil(float(min_frac) * int(total))))


def as_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "pass", "r"}
    return bool(v)


def coarse_decomposition(folds: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    alpha = float(rules["cert_alpha"])
    rows: list[dict[str, Any]] = []
    for (mode, h), g in folds.groupby(["return_mode", "horizon"], sort=True):
        total = int(g["fold"].nunique())
        req = required_support(total, int(rules["cert_min_folds"]), float(rules["cert_min_fold_fraction"]))
        p = pd.to_numeric(g.get("candidate_p_iut", np.nan), errors="coerce")
        ph = g.get("phase", pd.Series("", index=g.index)).astype(str)
        eligible = g.get("eligible", pd.Series(False, index=g.index)).map(as_bool)
        root_free = pd.to_numeric(g.get("train_n_roots", np.nan), errors="coerce").eq(0)
        holm = g.get("passes_horizon_holm", pd.Series(False, index=g.index)).map(as_bool)
        cert_r = g.get("certified_r_only_fold", pd.Series(False, index=g.index)).map(as_bool)
        local_r = eligible & root_free & ph.eq("R") & p.le(alpha)
        reasons = g.get("reason", pd.Series("", index=g.index)).fillna("").astype(str)
        rows.append({
            "return_mode": str(mode),
            "horizon": int(h),
            "folds_total": total,
            "folds_required": req,
            "root_free_folds": int(root_free.sum()),
            "eligible_folds": int(eligible.sum()),
            "phase_r_folds": int(ph.eq("R").sum()),
            "local_r_alpha_folds": int(local_r.sum()),
            "holm_pass_folds": int(holm.sum()),
            "certified_r_folds": int(cert_r.sum()),
            "enough_certified_r": bool(int(cert_r.sum()) >= req),
            "root_gate_fail_folds": int((~root_free).sum()),
            "insufficient_support_folds": int(reasons.eq("insufficient_oos_band_support").sum()),
            "bootstrap_unavailable_folds": int(reasons.eq("bootstrap_unavailable").sum()),
            "non_r_or_effect_ci_fail_folds": int((eligible & ~ph.eq("R")).sum()),
            "local_pass_but_holm_fail_folds": int((local_r & ~holm).sum()),
        })
    return pd.DataFrame(rows)


def refinement_decomposition(folds: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    alpha = float(rules["cert_alpha"])
    rows: list[dict[str, Any]] = []
    for (mode, h), g in folds.groupby(["return_mode", "horizon"], sort=True):
        total = int(g["fold"].nunique())
        req = required_support(total, int(rules["cert_min_folds"]), float(rules["cert_min_fold_fraction"]))
        p = pd.to_numeric(g.get("candidate_p_iut", np.nan), errors="coerce")
        ph = g.get("phase", pd.Series("", index=g.index)).astype(str)
        eligible = g.get("eligible", pd.Series(False, index=g.index)).map(as_bool)
        root_free = pd.to_numeric(g.get("train_n_roots", np.nan), errors="coerce").eq(0)
        local_flag = g.get("local_r_pass", pd.Series(False, index=g.index)).map(as_bool)
        fixed = g.get("fixed_sequence_r_pass", pd.Series(False, index=g.index)).map(as_bool)
        active = g.get("fixed_sequence_active", pd.Series(False, index=g.index)).map(as_bool)
        recomputed_local = eligible & root_free & ph.eq("R") & p.le(alpha)
        rows.append({
            "return_mode": str(mode),
            "horizon": int(h),
            "folds_total": total,
            "folds_required": req,
            "root_free_folds": int(root_free.sum()),
            "eligible_folds": int(eligible.sum()),
            "phase_r_folds": int(ph.eq("R").sum()),
            "local_r_folds": int(local_flag.sum()),
            "local_r_recomputed_folds": int(recomputed_local.sum()),
            "fixed_sequence_active_folds": int(active.sum()),
            "fixed_sequence_r_folds": int(fixed.sum()),
            "enough_fixed_sequence_r": bool(int(fixed.sum()) >= req),
            "local_flag_matches_recompute": bool((local_flag.to_numpy() == recomputed_local.to_numpy()).all()),
            "gate_stopped_folds": int((~active).sum()),
        })
    return pd.DataFrame(rows)


def compare_regimes(decomp: pd.DataFrame, regimes: pd.DataFrame, support_col: str) -> tuple[bool, list[dict[str, Any]]]:
    if decomp.empty or regimes.empty:
        return False, [{"reason": "missing_decomposition_or_regimes"}]
    mismatches: list[dict[str, Any]] = []
    rmap = {(str(r.return_mode), int(r.horizon)): str(r.certified_regime) for r in regimes.itertuples()}
    for r in decomp.itertuples():
        expected = "R" if int(getattr(r, support_col)) >= int(r.folds_required) else "UNRESOLVED"
        got = rmap.get((str(r.return_mode), int(r.horizon)))
        if got != expected and got != "M":
            mismatches.append({"return_mode": str(r.return_mode), "horizon": int(r.horizon), "expected": expected, "got": got})
    return len(mismatches) == 0, mismatches


def anchor_consistency(coarse_folds: pd.DataFrame, refine_folds: pd.DataFrame, upper: int) -> tuple[bool, list[dict[str, Any]]]:
    if coarse_folds.empty or refine_folds.empty:
        return False, [{"reason": "missing_fold_tables"}]
    mismatches: list[dict[str, Any]] = []
    for (mode, fold), cg in coarse_folds[coarse_folds["horizon"].astype(int).eq(int(upper))].groupby(["return_mode", "fold"]):
        rg = refine_folds[(refine_folds["return_mode"].astype(str) == str(mode)) & (refine_folds["fold"].astype(int) == int(fold)) & (refine_folds["horizon"].astype(int) == int(upper))]
        if rg.empty:
            mismatches.append({"return_mode": str(mode), "fold": int(fold), "reason": "missing_refinement_anchor_row"})
            continue
        cpass = as_bool(cg.iloc[0].get("certified_r_only_fold", False))
        rpass = as_bool(rg.iloc[0].get("fixed_sequence_r_pass", False))
        if cpass != rpass:
            mismatches.append({"return_mode": str(mode), "fold": int(fold), "coarse_anchor_pass": cpass, "refine_anchor_pass": rpass})
    return len(mismatches) == 0, mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only forensic audit of MRSPD independent confirmation")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--confirmatory-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    protocol_path = Path(args.protocol)
    root = Path(args.confirmatory_dir)
    out = Path(args.out) if args.out else root / "forensic-audit"
    out.mkdir(parents=True, exist_ok=True)

    protocol = read_json(protocol_path)
    pcopy = dict(protocol)
    expected_hash = pcopy.pop("protocol_sha256", None)
    hash_ok = expected_hash == canonical_sha256(pcopy)

    summary = read_json(root / "independent_validation_summary.json")
    independence = read_json(root / "independence_audit.json")
    hdesign = read_json(root / "confirmatory_horizon_design_audit.json")
    coarse_summary = read_json(root / "coarse" / "critical_horizon_summary.json")
    refine_summary = read_json(root / "refinement" / "hc_refinement_summary.json")

    coarse_folds = safe_read_csv(root / "coarse" / "critical_horizon_fold_certification.csv")
    coarse_regimes = safe_read_csv(root / "coarse" / "critical_horizon_regimes.csv")
    refine_folds = safe_read_csv(root / "refinement" / "hc_refinement_fold_certification.csv")
    refine_regimes = safe_read_csv(root / "refinement" / "hc_refinement_regimes.csv")
    reproduction = safe_read_csv(root / "refinement" / "hc_refinement_coarse_reproduction_audit.csv")

    rules = protocol["rules"]
    coarse_dec = coarse_decomposition(coarse_folds, rules)
    refine_dec = refinement_decomposition(refine_folds, rules)
    coarse_dec.to_csv(out / "coarse_failure_decomposition.csv", index=False)
    refine_dec.to_csv(out / "refinement_failure_decomposition.csv", index=False)

    coarse_regime_ok, coarse_regime_mismatch = compare_regimes(coarse_dec, coarse_regimes, "certified_r_folds")
    refine_regime_ok, refine_regime_mismatch = compare_regimes(refine_dec, refine_regimes, "fixed_sequence_r_folds")

    upper = int(protocol["refinement_interval"]["upper_inclusive"])
    anchor_ok, anchor_mismatches = anchor_consistency(coarse_folds, refine_folds, upper)
    reproduction_ok = bool(not reproduction.empty and reproduction.get("root_count_match", pd.Series(dtype=bool)).map(as_bool).all())

    surface = sorted(set(map(int, protocol["surface_fit_horizons"])))
    coarse_h = sorted(set(map(int, protocol["coarse_horizons"])))
    measurement = sorted(set(map(int, protocol["measurement_horizons"])))
    fine = sorted(set(map(int, protocol["refinement_horizons"])))
    horizon_design_recomputed = surface == coarse_h and set(coarse_h).union(fine).issubset(measurement) and not (set(fine) - set(coarse_h)).intersection(surface)

    assessment = summary.get("assessment", {})
    coarse_cons = coarse_summary.get("consensus_hc", {}) or {}
    refine_cons = refine_summary.get("consensus_hc_refined", {}) or {}
    target = int(protocol["discovery"]["target_hc_grid"])
    tol = int(protocol["primary_endpoints"]["localization_tolerance_days"])
    val_h = refine_cons.get("hc_grid")
    assessment_expected = {
        "independence_pass": bool(independence.get("pass")),
        "phenomenon_confirmed": bool(coarse_cons.get("identified")),
        "localization_confirmed": bool(refine_cons.get("identified") and val_h is not None and abs(int(val_h) - target) <= tol),
    }
    assessment_ok = all(bool(assessment.get(k)) == bool(v) for k, v in assessment_expected.items())

    invariant_checks = {
        "protocol_sha256_valid": bool(hash_ok),
        "independence_audit_pass": bool(independence.get("pass")),
        "horizon_design_file_pass": bool(hdesign.get("pass")),
        "horizon_design_recomputed_pass": bool(horizon_design_recomputed),
        "coarse_surface_root_reproduction_pass": bool(reproduction_ok),
        "coarse_regime_aggregation_consistent": bool(coarse_regime_ok),
        "refinement_regime_aggregation_consistent": bool(refine_regime_ok),
        "coarse_to_refinement_anchor_consistent": bool(anchor_ok),
        "assessment_recomputed_consistent": bool(assessment_ok),
    }
    implementation_ok = all(invariant_checks.values())

    def mode_blockers(dec: pd.DataFrame, support_col: str) -> list[dict[str, Any]]:
        outrows: list[dict[str, Any]] = []
        if dec.empty:
            return outrows
        for mode, g in dec.groupby("return_mode", sort=False):
            g = g.sort_values("horizon")
            passes = g[g[support_col] >= g["folds_required"]]
            first_pass = int(passes.iloc[0]["horizon"]) if not passes.empty else None
            best = g.sort_values([support_col, "horizon"], ascending=[False, True]).iloc[0]
            outrows.append({
                "return_mode": str(mode),
                "first_horizon_meeting_fold_support": first_pass,
                "best_horizon": int(best["horizon"]),
                "best_support": int(best[support_col]),
                "required_support": int(best["folds_required"]),
            })
        return outrows

    forensic = {
        "status": "MRSPD_CONFIRMATORY_FORENSIC_AUDIT_v8_4",
        "original_verdict_unchanged": assessment,
        "implementation_invariants_pass": bool(implementation_ok),
        "classification": "STATISTICAL_NONREPLICATION_UNDER_FROZEN_PROTOCOL" if implementation_ok and not bool(assessment.get("overall_confirmatory_pass")) else ("IMPLEMENTATION_MISMATCH_DETECTED" if not implementation_ok else "CONFIRMATION_PASS"),
        "invariant_checks": invariant_checks,
        "coarse_consensus": coarse_cons,
        "refined_consensus": refine_cons,
        "coarse_mode_support_summary": mode_blockers(coarse_dec, "certified_r_folds"),
        "refinement_mode_support_summary": mode_blockers(refine_dec, "fixed_sequence_r_folds"),
        "mismatches": {
            "coarse_regime": coarse_regime_mismatch[:50],
            "refinement_regime": refine_regime_mismatch[:50],
            "anchor": anchor_mismatches[:50],
        },
        "guardrail": "This audit is read-only. It does not change thresholds, horizons, folds, seeds, universe, or the original confirmatory verdict.",
    }
    (out / "forensic_summary.json").write_text(json.dumps(forensic, indent=2, default=str), encoding="utf-8")

    print(json.dumps(forensic, indent=2, default=str))
    print(f"FORENSIC_AUDIT: {'IMPLEMENTATION_OK' if implementation_ok else 'IMPLEMENTATION_MISMATCH'} -> {out}")


if __name__ == "__main__":
    main()

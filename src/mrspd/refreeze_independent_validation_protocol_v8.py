from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .independent_validation import PROTOCOL_VERSION, _canonical_json_sha256, _validate_protocol_horizon_design, freeze_validation_protocol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-freeze MRSPD independent-validation protocol after the v8 pre-validation refinement-method correction"
    )
    p.add_argument("--old-protocol", required=True)
    p.add_argument("--discovery-panel", required=True)
    p.add_argument("--coarse-summary", required=True)
    p.add_argument("--refinement-summary", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    old = json.loads(Path(args.old_protocol).read_text(encoding="utf-8"))
    old_hash = old.get("protocol_sha256")
    if not old_hash:
        raise ValueError("old protocol has no protocol_sha256")

    panel = pd.read_csv(Path(args.discovery_panel), parse_dates=["date"])
    coarse = json.loads(Path(args.coarse_summary).read_text(encoding="utf-8"))
    refine = json.loads(Path(args.refinement_summary).read_text(encoding="utf-8"))
    if "V8_METHOD_CORRECTION" not in str(refine.get("status", "")):
        raise ValueError("refinement summary is not a v8 method-corrected result")

    out = Path(args.out)
    protocol = freeze_validation_protocol(
        discovery_panel=panel,
        discovery_coarse_summary=coarse,
        discovery_refinement_summary=refine,
        out_path=out,
    )
    protocol.pop("protocol_sha256", None)
    protocol["protocol_version"] = PROTOCOL_VERSION
    protocol["status"] = "FROZEN_BEFORE_INDEPENDENT_VALIDATION_AFTER_V8_1_CONFIRMATORY_HORIZON_FIX"
    _validate_protocol_horizon_design(protocol)
    protocol["prevalidation_method_amendment"] = {
        "supersedes_protocol_sha256": str(old_hash),
        "reason": (
            "v7 dense-grid refinement changed the coarse spline fitting distribution/seed and used unordered Holm "
            "for an ordered localization problem. v8 freezes the exact v6 coarse fit and uses fixed-sequence "
            "gatekeeping from the pre-certified coarse upper anchor. v8.1 additionally separates surface_fit_horizons "
            "from measurement_horizons in the frozen confirmatory protocol and fail-closes the confirmation CLI against "
            "dense-fit contamination. No independent validation data had been read."
        ),
        "thresholds_changed": False,
        "alpha_changed": False,
        "effect_floor_changed": False,
        "bootstrap_changed": False,
        "fold_support_changed": False,
        "confirmatory_surface_fit_design_changed_from_buggy_protocol": True,
        "confirmatory_surface_fit_design": "exact v6 coarse horizons only",
        "dense_refinement_horizons_role": "measurement-only",
    }
    protocol.setdefault("guardrails", []).append(
        "This protocol supersedes the listed v7 hash before any independent validation data were analyzed."
    )
    protocol["protocol_sha256"] = _canonical_json_sha256(protocol)
    out.write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")
    print(json.dumps(protocol, indent=2, default=str))


if __name__ == "__main__":
    main()

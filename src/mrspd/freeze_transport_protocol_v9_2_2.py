from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .robust_transport_v9_2 import PROTOCOL_VERSION, HUBER_C, MAD_NORMALIZER

LOCK_VERSION = "MRSPD-ROBUST-PERCENTILE-TRANSPORT-v9.2.2-CONFIRMATORY-LOCK"

def canonical_hash(doc: dict) -> str:
    x = dict(doc)
    x.pop("protocol_sha256", None)
    raw = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()

def _local_resolution(horizons: list[int], lo: int, hi: int) -> int:
    H = sorted(set(map(int, horizons)))
    diffs = []
    for a, b in zip(H[:-1], H[1:]):
        if b >= lo and a <= hi:
            diffs.append(b-a)
    if not diffs:
        raise ValueError("cannot determine local horizon resolution")
    return int(min(diffs))

def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the selected v9.2 robust transport hypothesis before any V3 price inspection")
    ap.add_argument("--exploratory-summary", required=True)
    ap.add_argument("--selected-hypothesis", required=True)
    ap.add_argument("--discovery-panel", required=True)
    ap.add_argument("--second-panel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sp = Path(a.exploratory_summary); hp = Path(a.selected_hypothesis)
    summary = json.loads(sp.read_text(encoding="utf-8"))
    selected = json.loads(hp.read_text(encoding="utf-8"))
    if summary.get("status") != "EXPLORATORY_TRANSPORT_V9_2_ROBUST_DELAYED":
        raise ValueError("wrong exploratory summary type")
    if summary.get("selected_transport_hypothesis") != selected:
        raise ValueError("selected hypothesis file does not match exploratory summary")
    if not selected.get("selected"):
        raise SystemExit("REFUSE_TO_FREEZE: no v9.2 development candidate selected")
    if selected.get("normalization") != "global_percentile":
        raise ValueError("v9.2.2 lock only supports global_percentile")
    if selected.get("response") != "cross_sectional_huber_clipped_phase_product":
        raise ValueError("unexpected v9.2 response")
    if abs(float(selected.get("huber_c")) - HUBER_C) > 1e-12:
        raise ValueError("Huber c mismatch")

    rules = summary["rules"]
    H = list(map(int, summary["horizons"]))
    dh = int(selected["discovery_hc_grid"]); sh = int(selected["second_hc_grid"])
    lo, hi = min(dh, sh), max(dh, sh)
    development_delta = abs(dh-sh)
    local_resolution = _local_resolution(H, lo, hi)
    tolerance = int(max(development_delta, local_resolution))

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "lock_version": LOCK_VERSION,
        "status": "FROZEN_BEFORE_THIRD_UNIVERSE_PRICE_INSPECTION",
        "development_sources": {
            "exploratory_summary_sha256": file_sha256(sp),
            "selected_hypothesis_sha256": file_sha256(hp),
            "discovery_panel_sha256": file_sha256(Path(a.discovery_panel)),
            "second_panel_sha256": file_sha256(Path(a.second_panel)),
        },
        "hypothesis": {
            "coordinate": "global empirical q_psi percentile within universe x date x return_mode x horizon",
            "response": "median + clip(phase_product-median, +/- c*1.4826*MAD) within date x return_mode x horizon",
            "tail_direction": "upper",
            "q0": float(selected["q0"]),
            "tail_bins": 2,
            "huber_c": float(HUBER_C),
            "mad_normalizer": float(MAD_NORMALIZER),
            "claim": "persistent OOS negative robust phase response in both upper-tail sub-bands for raw and market_residual",
        },
        "horizons": H,
        "rules": rules,
        "development_localization": {
            "discovery_hc_grid": dh,
            "second_hc_grid": sh,
            "expected_hc_envelope": [lo, hi],
            "development_hc_delta_days": development_delta,
            "local_grid_resolution_days": local_resolution,
        },
        "third_universe_pass_rule": {
            "independence_required": True,
            "bound_ticker_subset_required": True,
            "minimum_bound_ticker_coverage_fraction": 0.80,
            "both_return_modes_required": True,
            "persistent_tail_required": True,
            "expected_hc_envelope": [lo, hi],
            "localization_tolerance_days": tolerance,
        },
        "amendment_note": "New post-v8/v9 development hypothesis. Earlier negative confirmatory verdicts remain unchanged.",
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "FROZEN_V9_2_2",
        "q0": protocol["hypothesis"]["q0"],
        "expected_hc_envelope": [lo, hi],
        "localization_tolerance_days": tolerance,
        "protocol_sha256": protocol["protocol_sha256"],
        "out": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import pandas as pd

from .robust_transport_v9_2 import (
    HUBER_C, MAD_NORMALIZER,
    load_panel_streaming, add_global_percentile_and_robust_response,
    certify_candidate, aggregate_regimes, identify_hc, consensus_for_q0,
)

CHUNK = 250_000

def norm(x): return str(x).strip().upper()

def canonical_sha(doc):
    x = dict(doc); x.pop("protocol_sha256", None)
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def sha_set(xs):
    return hashlib.sha256("\n".join(sorted(set(norm(x) for x in xs))).encode("utf-8")).hexdigest()

def panel_meta(path):
    tickers, horizons, modes = set(), set(), set()
    rows = 0
    for c in pd.read_csv(path, usecols=["ticker","horizon","return_mode"], chunksize=CHUNK):
        rows += len(c)
        tickers.update(norm(x) for x in c["ticker"].dropna().unique())
        horizons.update(int(x) for x in c["horizon"].dropna().unique())
        modes.update(str(x) for x in c["return_mode"].dropna().unique())
    return {"rows": rows, "tickers": tickers, "horizons": horizons, "modes": modes}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--development-panel", action="append", required=True)
    ap.add_argument("--v4-panel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_AND_BOUND_V10_BEFORE_V4_PANEL_CONSTRUCTION":
        raise ValueError("use bound v10 V4 protocol")
    if p.get("protocol_sha256") != canonical_sha(p):
        raise ValueError("protocol SHA mismatch")

    design = p["v4_design"]
    bound = set(norm(x) for x in design["tickers"])
    if design["ticker_set_sha256"] != sha_set(bound):
        raise ValueError("V4 ticker-set SHA mismatch")

    v4_path = Path(a.v4_panel)
    manifest_path = v4_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing V4 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_market = design["market_benchmark"]["ticker"]
    observed_market = str((manifest.get("config") or {}).get("market_ticker","")).strip()
    if observed_market != expected_market:
        raise ValueError(f"V4 market benchmark mismatch protocol={expected_market} manifest={observed_market}")

    vm = panel_meta(v4_path)
    used = set()
    for dp in a.development_panel:
        used |= panel_meta(Path(dp))["tickers"]

    outside = sorted(vm["tickers"] - bound)
    overlap = sorted(vm["tickers"] & used)
    coverage = len(vm["tickers"]) / max(1, len(bound))
    horizons = list(map(int, p["horizons"]))
    missing_h = sorted(set(horizons) - vm["horizons"])
    modes_ok = {"raw","market_residual"}.issubset(vm["modes"])

    design_audit = {
        "market_benchmark_pass": observed_market == expected_market,
        "independence_pass": len(overlap) == 0,
        "bound_subset_pass": len(outside) == 0,
        "coverage_fraction": coverage,
        "coverage_pass": coverage >= float(design["minimum_panel_coverage_fraction"]),
        "horizon_design_pass": len(missing_h) == 0,
        "return_modes_present_pass": modes_ok,
        "v4_panel_ticker_count": len(vm["tickers"]),
        "bound_ticker_count": len(bound),
        "overlap_count": len(overlap),
        "outside_bound_count": len(outside),
        "missing_horizons": missing_h,
    }

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/"v4_design_audit.json").write_text(json.dumps(design_audit, indent=2), encoding="utf-8")
    if not all(design_audit[k] for k in [
        "market_benchmark_pass","independence_pass","bound_subset_pass",
        "coverage_pass","horizon_design_pass","return_modes_present_pass"
    ]):
        raise ValueError(f"V4 design audit failed: {design_audit}")

    hyp, rules = p["hypothesis"], p["rules"]
    if abs(float(hyp["q0"]) - 0.85) > 1e-12:
        raise ValueError("q0 mismatch")
    if abs(float(hyp["huber_c"]) - HUBER_C) > 1e-12 or abs(float(hyp["mad_normalizer"]) - MAD_NORMALIZER) > 1e-12:
        raise ValueError("robust constants mismatch")

    raw = load_panel_streaming(v4_path, horizons=horizons)
    pre = add_global_percentile_and_robust_response(raw, huber_c=HUBER_C)
    del raw
    folds = certify_candidate(
        pre, universe="v4_confirmatory", q0=float(hyp["q0"]), horizons=horizons,
        splits=int(rules["walkforward_splits"]), min_train_frac=float(rules["min_train_frac"]),
        bootstrap_reps=int(rules["bootstrap_reps"]), alpha=float(rules["alpha"]),
        min_effect_sigma=float(rules["min_effect_sigma"]),
        min_obs_per_bin=int(rules["min_obs_per_tail_bin"]),
        min_dates_per_bin=int(rules["min_dates_per_tail_bin"]),
        min_train_dates_per_bin=int(rules["min_train_dates_per_tail_bin"]),
        seed=int(rules["seed"]),
    )
    regs = aggregate_regimes(
        folds, min_folds=int(rules["min_folds"]),
        min_fold_fraction=float(rules["min_fold_fraction"])
    )
    hc_rows = identify_hc(
        regs, horizons, min_consecutive=int(rules["hc_min_consecutive"]),
        min_tail_fraction=float(rules["hc_min_tail_fraction"])
    )
    cons = consensus_for_q0(hc_rows, "v4_confirmatory", float(hyp["q0"]))

    folds.to_csv(out/"v4_fold_certification.csv", index=False)
    regs.to_csv(out/"v4_regimes.csv", index=False)
    pd.DataFrame(hc_rows).to_csv(out/"v4_hc.csv", index=False)

    mode_hc = {}
    for row in hc_rows:
        if str(row.get("universe")) == "v4_confirmatory":
            mode_hc[str(row.get("return_mode"))] = (
                int(row["hc_grid"]) if row.get("identified") and row.get("hc_grid") is not None else None
            )
    raw_ok = mode_hc.get("raw") is not None
    residual_ok = mode_hc.get("market_residual") is not None
    overall = bool(raw_ok and residual_ok)

    summary = {
        "status": "V10_V4_CONFIRMATORY_VALIDATION",
        "design_audit_pass": True,
        "raw_hc_identified": raw_ok,
        "market_residual_hc_identified": residual_ok,
        "mode_hc": mode_hc,
        "consensus": cons,
        "localization_gate": None,
        "overall_confirmatory_pass": overall,
        "interpretation": "CONFIRMED" if overall else "NOT_CONFIRMED",
        "protocol_sha256": p["protocol_sha256"],
        "guardrail": (
            "The v10 primary endpoint tests cross-market existence of the persistent robust reversal tail. "
            "Market-specific h_c is estimated OOS and is not forced to match US or UK development locations."
        ),
    }
    (out/"v10_v4_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

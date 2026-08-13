from __future__ import annotations

import argparse
import hashlib
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

CHUNK = 250_000

ALLOWED_BOUND_STATUSES = {
    "FROZEN_AND_BOUND_BEFORE_THIRD_UNIVERSE_PANEL_CONSTRUCTION",
    "BOUND_V9_2_3_WITH_LOCAL_MARKET_BENCHMARK",
}


def norm(x: object) -> str:
    # Preserve exchange suffixes such as Yahoo Finance London ".L".
    # Do NOT use the old US-class-share normalization .replace(".", "-").
    return str(x).strip().upper()


def canonical_hash(doc: dict) -> str:
    x = dict(doc)
    x.pop("protocol_sha256", None)
    return hashlib.sha256(
        json.dumps(
            x,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def ticker_set_sha256(tickers: list[str] | set[str]) -> str:
    xs = sorted(set(norm(x) for x in tickers))
    return hashlib.sha256("\n".join(xs).encode("utf-8")).hexdigest()


def validate_bound_protocol(p: dict) -> None:
    status = str(p.get("status", ""))
    if status not in ALLOWED_BOUND_STATUSES:
        raise ValueError(
            "use a bound v9.2 protocol; "
            f"unsupported status={status!r}; "
            f"allowed={sorted(ALLOWED_BOUND_STATUSES)}"
        )

    design = p.get("third_universe_design")
    if not isinstance(design, dict):
        raise ValueError("bound protocol missing third_universe_design")
    if design.get("binding_status") != "BOUND_BEFORE_V3_PANEL_CONSTRUCTION":
        raise ValueError(
            "third-universe binding status mismatch: "
            f"{design.get('binding_status')!r}"
        )

    bound = sorted(set(norm(x) for x in design.get("tickers", [])))
    if not bound:
        raise ValueError("bound protocol contains an empty third-universe ticker set")

    stored_tsha = design.get("ticker_set_sha256")
    actual_tsha = ticker_set_sha256(bound)
    if stored_tsha != actual_tsha:
        raise ValueError(
            "bound third-universe ticker SHA256 mismatch: "
            f"stored={stored_tsha} actual={actual_tsha}"
        )

    if int(design.get("ticker_count", -1)) != len(bound):
        raise ValueError(
            "bound third-universe ticker_count mismatch: "
            f"stored={design.get('ticker_count')} actual={len(bound)}"
        )


def audit_frozen_market_benchmark(protocol: dict, third_panel: Path) -> dict:
    """
    If the protocol contains an outcome-blind frozen local-market benchmark,
    require the panel's sibling manifest.json to prove that exact ticker was
    used to build market_residual.
    """
    frozen = protocol.get("third_universe_market_benchmark")
    if frozen is None:
        return {
            "required": False,
            "pass": True,
            "frozen_market_ticker": None,
            "panel_market_ticker": None,
            "manifest": None,
        }

    expected = str(frozen.get("ticker", "")).strip()
    if not expected:
        raise ValueError("third_universe_market_benchmark has empty ticker")

    manifest_path = third_panel.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "local-market benchmark is frozen but sibling manifest.json is missing: "
            f"{manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = str(
        (manifest.get("config") or {}).get("market_ticker", "")
    ).strip()

    passed = observed == expected
    audit = {
        "required": True,
        "pass": passed,
        "frozen_market_ticker": expected,
        "panel_market_ticker": observed,
        "manifest": str(manifest_path),
    }
    if not passed:
        raise ValueError(
            "third-universe market benchmark mismatch: "
            f"protocol={expected!r} panel_manifest={observed!r}"
        )
    return audit


def panel_meta(path: Path) -> dict:
    tickers = set()
    horizons = set()
    modes = set()
    n = 0
    dmin = None
    dmax = None

    for c in pd.read_csv(
        path,
        usecols=["date", "ticker", "return_mode", "horizon"],
        chunksize=CHUNK,
    ):
        n += len(c)
        tickers.update(norm(x) for x in c["ticker"].dropna().unique())
        horizons.update(int(x) for x in c["horizon"].dropna().unique())
        modes.update(str(x) for x in c["return_mode"].dropna().unique())

        d = pd.to_datetime(c["date"], errors="coerce")
        if d.notna().any():
            lo, hi = d.min(), d.max()
            dmin = lo if dmin is None or lo < dmin else dmin
            dmax = hi if dmax is None or hi > dmax else dmax

    return {
        "rows": n,
        "tickers": sorted(tickers),
        "ticker_count": len(tickers),
        "horizons": sorted(horizons),
        "return_modes": sorted(modes),
        "date_min": str(dmin.date()) if dmin is not None else None,
        "date_max": str(dmax.date()) if dmax is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Locked V3 confirmatory validation for MRSPD robust percentile "
            "transport v9.2"
        )
    )
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--discovery-panel", required=True)
    ap.add_argument("--second-panel", required=True)
    ap.add_argument("--third-panel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pp = Path(a.protocol)
    p = json.loads(pp.read_text(encoding="utf-8"))

    if p.get("protocol_sha256") != canonical_hash(p):
        raise ValueError("protocol SHA256 mismatch")

    validate_bound_protocol(p)

    hyp = p["hypothesis"]
    rules = p["rules"]
    horizons = list(map(int, p["horizons"]))
    q0 = float(hyp["q0"])

    if hyp.get("tail_direction") != "upper" or int(hyp.get("tail_bins")) != 2:
        raise ValueError("unsupported frozen tail definition")

    if (
        abs(float(hyp["huber_c"]) - HUBER_C) > 1e-12
        or abs(float(hyp["mad_normalizer"]) - MAD_NORMALIZER) > 1e-12
    ):
        raise ValueError("robust-response constants mismatch")

    third_panel = Path(a.third_panel)
    benchmark_audit = audit_frozen_market_benchmark(p, third_panel)

    dm = panel_meta(Path(a.discovery_panel))
    sm = panel_meta(Path(a.second_panel))
    tm = panel_meta(third_panel)

    used = set(dm["tickers"]) | set(sm["tickers"])
    third = set(tm["tickers"])
    overlap = sorted(used & third)

    bound = set(norm(x) for x in p["third_universe_design"]["tickers"])
    outside = sorted(third - bound)
    coverage = len(third) / max(1, len(bound))
    mincov = float(
        p["third_universe_design"]["min_panel_coverage_fraction"]
    )

    missing_h = sorted(set(horizons) - set(tm["horizons"]))
    modes_ok = {"raw", "market_residual"}.issubset(set(tm["return_modes"]))

    audit = {
        "protocol_status_pass": True,
        "protocol_sha256_pass": True,
        "ticker_binding_sha256_pass": True,
        "market_benchmark_audit": benchmark_audit,
        "market_benchmark_pass": bool(benchmark_audit["pass"]),
        "independence_pass": not overlap,
        "bound_subset_pass": not outside,
        "coverage_pass": coverage >= mincov,
        "horizon_design_pass": not missing_h,
        "return_modes_present_pass": modes_ok,
        "third_ticker_count": len(third),
        "bound_ticker_count": len(bound),
        "coverage_fraction": coverage,
        "minimum_coverage_fraction": mincov,
        "overlap_count": len(overlap),
        "overlap_preview": overlap[:20],
        "outside_bound_count": len(outside),
        "outside_bound_preview": outside[:20],
        "missing_horizons": missing_h,
        "third_dates": [tm["date_min"], tm["date_max"]],
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "third_universe_v9_2_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    required_audit_keys = [
        "market_benchmark_pass",
        "independence_pass",
        "bound_subset_pass",
        "coverage_pass",
        "horizon_design_pass",
        "return_modes_present_pass",
    ]
    if not all(audit[k] for k in required_audit_keys):
        raise ValueError(f"V3 design audit failed: {audit}")

    raw = load_panel_streaming(third_panel, horizons=horizons)
    pre = add_global_percentile_and_robust_response(
        raw,
        huber_c=HUBER_C,
    )
    del raw

    folds = certify_candidate(
        pre,
        universe="third_universe",
        q0=q0,
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

    cons = consensus_for_q0(
        hc_rows,
        "third_universe",
        q0,
    )

    folds.to_csv(
        out / "third_robust_tail_fold_certification.csv",
        index=False,
    )
    regs.to_csv(
        out / "third_robust_tail_regimes.csv",
        index=False,
    )
    pd.DataFrame(hc_rows).to_csv(
        out / "third_robust_tail_hc.csv",
        index=False,
    )

    phenomenon = bool(cons.get("identified"))
    vh = cons.get("hc_grid")

    pass_rule = p["third_universe_pass_rule"]
    env = list(map(int, pass_rule["expected_hc_envelope"]))
    tol = int(pass_rule["localization_tolerance_days"])

    localization = bool(
        phenomenon
        and vh is not None
        and env[0] - tol <= int(vh) <= env[1] + tol
    )

    verdict = {
        "independence_pass": True,
        "bound_universe_pass": True,
        "market_benchmark_pass": True,
        "phenomenon_confirmed": phenomenon,
        "localization_confirmed": localization,
        "validation_hc_grid": int(vh) if vh is not None else None,
        "frozen_q0": q0,
        "frozen_response": "cross_sectional_huber_clipped_phase_product",
        "frozen_expected_hc_envelope": env,
        "localization_tolerance_days": tol,
        "overall_confirmatory_pass": bool(phenomenon and localization),
        "interpretation": (
            "CONFIRMED"
            if phenomenon and localization
            else "NOT_CONFIRMED"
        ),
        "consensus": cons,
        "protocol_sha256": p["protocol_sha256"],
        "third_universe_market_benchmark": p.get(
            "third_universe_market_benchmark"
        ),
    }

    (out / "third_universe_validation_summary_v9_2.json").write_text(
        json.dumps(verdict, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()

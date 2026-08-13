#!/usr/bin/env python3
"""
Outcome-blind amendment of an already bound MRSPD v9.2 protocol:
freeze the third-universe market benchmark before any V3 price analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha(doc: dict) -> str:
    x = dict(doc)
    x.pop("protocol_sha256", None)
    payload = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--market-ticker", required=True)
    ap.add_argument("--market-name", required=True)
    ap.add_argument("--market-scope", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.protocol)
    out = Path(a.out)

    p = json.loads(src.read_text(encoding="utf-8"))
    status = str(p.get("status", ""))
    if "BOUND" not in status:
        raise SystemExit(f"MARKET_BENCHMARK_BIND: FAIL protocol is not bound; status={status!r}")

    ticker = a.market_ticker.strip()
    if not ticker:
        raise SystemExit("MARKET_BENCHMARK_BIND: FAIL empty market ticker")

    # Do not silently overwrite an already-frozen market benchmark.
    existing = p.get("third_universe_market_benchmark")
    if existing is not None:
        raise SystemExit(
            "MARKET_BENCHMARK_BIND: FAIL benchmark already frozen in protocol: "
            + json.dumps(existing, ensure_ascii=False)
        )

    p["status"] = "BOUND_V9_2_3_WITH_LOCAL_MARKET_BENCHMARK"
    p["third_universe_market_benchmark"] = {
        "ticker": ticker,
        "name": a.market_name,
        "scope": a.market_scope,
        "residual_definition": "raw_log_return - rolling_beta(raw_log_return, market_log_return) * market_log_return",
        "decision_timing": "frozen_after_ticker_binding_and_before_third_universe_price_analysis",
    }
    p["cross_market_rule"] = {
        "principle": "market_residual uses a local broad equity-market benchmark appropriate to each universe",
        "development_market_benchmark": "SPY",
        "third_universe_market_benchmark": ticker,
        "raw_mode_unchanged": True,
    }
    p["parent_protocol_sha256"] = p.get("protocol_sha256")
    p["protocol_sha256"] = canonical_sha(p)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": p["status"],
        "market_ticker": ticker,
        "market_name": a.market_name,
        "parent_protocol_sha256": p.get("parent_protocol_sha256"),
        "protocol_sha256": p["protocol_sha256"],
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

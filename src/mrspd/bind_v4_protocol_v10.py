#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import pandas as pd

def norm(x): return str(x).strip().upper()

def sha_set(xs):
    vals = sorted(set(norm(x) for x in xs))
    return hashlib.sha256("\n".join(vals).encode("utf-8")).hexdigest()

def canonical_sha(doc):
    x = dict(doc); x.pop("protocol_sha256", None)
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def panel_tickers(path):
    s = set()
    for c in pd.read_csv(path, usecols=["ticker"], chunksize=250_000):
        s.update(norm(x) for x in c["ticker"].dropna().unique())
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--v4-universe", required=True)
    ap.add_argument("--development-panel", action="append", required=True)
    ap.add_argument("--market-ticker", required=True)
    ap.add_argument("--market-name", required=True)
    ap.add_argument("--market-scope", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_V10_BEFORE_V4_PANEL_CONSTRUCTION":
        raise SystemExit("V4_BIND: FAIL use frozen v10 protocol")
    if p.get("protocol_sha256") != canonical_sha(p):
        raise SystemExit("V4_BIND: FAIL protocol SHA mismatch")

    u = pd.read_csv(a.v4_universe)
    if "ticker" not in u.columns:
        raise SystemExit("V4_BIND: FAIL V4 universe missing ticker")
    tickers = sorted(set(norm(x) for x in u["ticker"].dropna()))
    if not tickers:
        raise SystemExit("V4_BIND: FAIL empty V4 ticker list")

    used = set()
    for dp in a.development_panel:
        used |= panel_tickers(dp)
    overlap = sorted(set(tickers) & used)
    if overlap:
        raise SystemExit(f"V4_BIND: FAIL overlap with development: {overlap[:20]}")

    p["status"] = "FROZEN_AND_BOUND_V10_BEFORE_V4_PANEL_CONSTRUCTION"
    p["v4_design"] = {
        "tickers": tickers,
        "ticker_count": len(tickers),
        "ticker_set_sha256": sha_set(tickers),
        "minimum_panel_coverage_fraction": float(p["v4_guardrails"]["minimum_panel_coverage_fraction"]),
        "binding_status": "BOUND_BEFORE_V4_PANEL_CONSTRUCTION",
        "no_replacements_after_binding": True,
        "market_benchmark": {
            "ticker": a.market_ticker.strip(),
            "name": a.market_name,
            "scope": a.market_scope,
        },
    }
    p["parent_frozen_v10_protocol_sha256"] = p.get("protocol_sha256")
    p["protocol_sha256"] = canonical_sha(p)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": p["status"],
        "v4_ticker_count": len(tickers),
        "v4_ticker_set_sha256": p["v4_design"]["ticker_set_sha256"],
        "market_ticker": p["v4_design"]["market_benchmark"]["ticker"],
        "protocol_sha256": p["protocol_sha256"],
        "out": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()

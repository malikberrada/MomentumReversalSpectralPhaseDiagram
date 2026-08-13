#!/usr/bin/env python3
"""
Small preflight wrapper for bind_v4_protocol_v10.py.
It fails clearly on empty/malformed candidate CSV before invoking the existing binder.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder", default="bind_v4_protocol_v10.py")
    ap.add_argument("--v4-universe", required=True)
    args, rest = ap.parse_known_args()

    p = Path(args.v4_universe)
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit(
            f"V4_BIND_PREFLIGHT: FAIL {p} is missing or empty. "
            "Run make_v4_candidate_nikkei225_v10_1.py first."
        )
    try:
        u = pd.read_csv(p)
    except EmptyDataError:
        raise SystemExit(
            f"V4_BIND_PREFLIGHT: FAIL {p} contains no CSV columns."
        )

    if "ticker" not in u.columns:
        raise SystemExit("V4_BIND_PREFLIGHT: FAIL candidate CSV missing ticker column")
    if len(u) != 225:
        raise SystemExit(
            f"V4_BIND_PREFLIGHT: FAIL expected 225 frozen candidates, found {len(u)}"
        )
    if not u["ticker"].astype(str).str.endswith(".T").all():
        raise SystemExit("V4_BIND_PREFLIGHT: FAIL one or more tickers are not Tokyo .T symbols")

    cmd = [sys.executable, args.binder, "--v4-universe", str(p)] + rest
    print("V4_BIND_PREFLIGHT: PASS; invoking binder")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

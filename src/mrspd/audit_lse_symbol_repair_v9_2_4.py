#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha_text(xs: list[str]) -> str:
    return hashlib.sha256("\n".join(xs).encode("utf-8")).hexdigest()


def issuer_key(x: object) -> str:
    s = str(x).strip().upper()
    if s.endswith(".L"):
        return s[:-2]
    if s.endswith("-L"):
        return s[:-2]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit deterministic MRSPD V3 LSE symbol-notation repair (-L -> .L) without using prices."
    )
    ap.add_argument("--old-universe", required=True)
    ap.add_argument("--new-universe", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    old = pd.read_csv(a.old_universe)
    new = pd.read_csv(a.new_universe)
    if "ticker" not in old.columns or "ticker" not in new.columns:
        raise ValueError("both universes must contain ticker")

    ot = sorted(set(old["ticker"].dropna().astype(str).str.strip().str.upper()))
    nt = sorted(set(new["ticker"].dropna().astype(str).str.strip().str.upper()))
    ok = sorted(issuer_key(x) for x in ot)
    nk = sorted(issuer_key(x) for x in nt)

    one_to_one = (
        len(ot) == len(nt)
        and len(set(ok)) == len(ok)
        and len(set(nk)) == len(nk)
        and ok == nk
    )

    malformed_old = sum(x.endswith("-L") for x in ot)
    proper_new = sum(x.endswith(".L") for x in nt)

    audit = {
        "status": "LSE_SYMBOL_NOTATION_REPAIR_AUDIT",
        "old_ticker_count": len(ot),
        "new_ticker_count": len(nt),
        "old_terminal_dash_L_count": malformed_old,
        "new_terminal_dot_L_count": proper_new,
        "issuer_identity_one_to_one": bool(one_to_one),
        "issuer_set_sha256": sha_text(ok),
        "old_ticker_set_sha256": sha_text(ot),
        "new_ticker_set_sha256": sha_text(nt),
        "scientific_interpretation": (
            "symbol-notation-only implementation repair; no issuer was added, removed, or selected using V3 outcomes"
            if one_to_one
            else "FAIL: issuer sets differ"
        ),
        "pass": bool(one_to_one and malformed_old == len(ot) and proper_new == len(nt)),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

    if not audit["pass"]:
        raise SystemExit("LSE_SYMBOL_REPAIR_AUDIT: FAIL")
    print("LSE_SYMBOL_REPAIR_AUDIT: PASS")


if __name__ == "__main__":
    main()

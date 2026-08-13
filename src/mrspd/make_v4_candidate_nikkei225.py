#!/usr/bin/env python3
from __future__ import annotations

import argparse, io, re
from pathlib import Path

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/Nikkei_225"
UA = "Mozilla/5.0 MRSPD-v10-research/1.0"

def flatten(df):
    x = df.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [" ".join(str(v).strip() for v in tup if str(v) != "nan").strip() for tup in x.columns]
    else:
        x.columns = [str(c).strip() for c in x.columns]
    return x

def find_col(cols, needles):
    for c in cols:
        lc = str(c).lower()
        if any(n in lc for n in needles):
            return c
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Build V4 candidate ticker list from the Nikkei 225 constituent table only; no prices are read."
    )
    ap.add_argument("--out", default="universe_candidate_v4_japan.csv")
    ap.add_argument("--min-candidates", type=int, default=220)
    a = ap.parse_args()

    r = requests.get(URL, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))

    candidates = []
    for t in tables:
        t = flatten(t)
        code_col = find_col(t.columns, ["code"])
        company_col = find_col(t.columns, ["company", "name"])
        if code_col is None or company_col is None:
            continue
        codes = t[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
        valid = codes.notna()
        if valid.sum() < 150:
            continue
        sector_col = find_col(t.columns, ["sector", "industry"])
        x = pd.DataFrame({
            "ticker": codes[valid].str.zfill(4) + ".T",
            "asset_class": "equity",
            "source_index": "nikkei225",
            "sector": (
                t.loc[valid, sector_col].astype(str).str.strip().values
                if sector_col is not None else ["UNKNOWN"] * int(valid.sum())
            ),
            "company": t.loc[valid, company_col].astype(str).str.strip().values,
            "source_url": URL,
        })
        candidates.append(x)

    if not candidates:
        raise SystemExit("V4_CANDIDATE_BUILD: FAIL no Nikkei 225 component table parsed")
    out = max(candidates, key=len).drop_duplicates("ticker").reset_index(drop=True)
    if len(out) < a.min_candidates:
        raise SystemExit(f"V4_CANDIDATE_BUILD: FAIL only {len(out)} unique tickers")
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"V4_CANDIDATE_BUILD: PASS tickers={len(out)} -> {path}")

if __name__ == "__main__":
    main()

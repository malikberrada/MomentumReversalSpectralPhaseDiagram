#!/usr/bin/env python3
"""
Build an untouched third-universe candidate list for MRSPD v9.2 from
FTSE 100 + FTSE 250 constituent tables.

This script fetches constituent TABLES only. It does not download or inspect
price/return data.

Output columns:
    ticker, asset_class, source_index, sector, company, source_url
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

SOURCES = [
    ("ftse100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
    ("ftse250", "https://en.wikipedia.org/wiki/FTSE_250_Index"),
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36 MRSPD-research/1.0"
)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(x).strip() for x in tup if str(x) != "nan").strip()
            for tup in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def find_col(columns, candidates):
    norm = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in columns}
    for cand in candidates:
        cand = cand.lower()
        for k, original in norm.items():
            if k == cand or cand in k:
                return original
    return None


def yahoo_lse_symbol(epic: object) -> str | None:
    if pd.isna(epic):
        return None
    s = str(epic).strip().upper()
    if not s or s in {"NAN", "—", "-", "–"}:
        return None

    # Remove footnote markers / spaces.
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\s+", "", s)

    # LSE EPICs can contain a trailing dot (e.g. "RR."). Yahoo's market
    # suffix supplies the dot, so remove trailing punctuation first.
    s = s.rstrip(".")

    # For class shares, Yahoo commonly represents an internal dot as "-".
    s = s.replace(".", "-")

    # Keep only conservative Yahoo-symbol characters.
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    if not s:
        return None
    return f"{s}.L"


def fetch_constituents(source_name: str, url: str) -> pd.DataFrame:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()

    tables = pd.read_html(io.StringIO(r.text))
    matches = []

    for t in tables:
        t = flatten_columns(t)
        ticker_col = find_col(t.columns, ["ticker", "epic"])
        company_col = find_col(t.columns, ["company", "constituent"])
        if ticker_col is None or company_col is None:
            continue

        sector_col = find_col(
            t.columns,
            [
                "ftse industry classification benchmark sector",
                "industry classification benchmark sector",
                "sector",
            ],
        )

        x = pd.DataFrame(
            {
                "company": t[company_col].astype(str).str.strip(),
                "epic": t[ticker_col],
                "sector": (
                    t[sector_col].astype(str).str.strip()
                    if sector_col is not None
                    else "UNKNOWN"
                ),
            }
        )
        x["ticker"] = x["epic"].map(yahoo_lse_symbol)
        x = x[x["ticker"].notna()].copy()
        x = x[
            ~x["company"].str.lower().isin({"nan", "company", "constituent"})
        ]
        x["asset_class"] = "equity"
        x["source_index"] = source_name
        x["source_url"] = url

        if len(x) >= 40:
            matches.append(x)

    if not matches:
        raise RuntimeError(
            f"No constituent table found for {source_name} at {url}. "
            "The webpage table layout may have changed."
        )

    # Prefer the largest plausible constituent table.
    x = max(matches, key=len)
    return x[
        ["ticker", "asset_class", "source_index", "sector", "company", "source_url"]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="universe_candidate_third.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--min-candidates",
        type=int,
        default=300,
        help="Fail if fewer than this many unique candidate tickers are obtained.",
    )
    args = ap.parse_args()

    frames = []
    for name, url in SOURCES:
        x = fetch_constituents(name, url)
        print(f"{name}: {len(x):,} parsed rows")
        frames.append(x)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker"], keep="first")
    out = out.sort_values(["source_index", "ticker"]).reset_index(drop=True)

    if len(out) < args.min_candidates:
        raise SystemExit(
            f"THIRD_CANDIDATE_BUILD: FAIL: only {len(out)} unique tickers; "
            f"minimum required is {args.min_candidates}"
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print(f"candidate tickers: {len(out):,}")
    print(f"output: {path}")
    print("THIRD_CANDIDATE_BUILD: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

OFFICIAL_URL = "https://indexes.nikkei.co.jp/en/nkave/index/component"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36 MRSPD-v10-research/1.0"
)

CODE_RE = re.compile(r"^(?:\d{4}|\d{3}[A-Z])$")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(x).strip() for x in tup if str(x).lower() != "nan").strip()
            for tup in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def normalized_name(x: object) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def find_exactish(columns, wanted):
    wanted = wanted.lower()
    for c in columns:
        k = re.sub(r"\s+", " ", str(c).strip().lower())
        if k == wanted or wanted in k:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build an outcome-blind MRSPD V4 candidate universe from the "
            "official Nikkei 225 component page. Constituents only; no prices."
        )
    )
    ap.add_argument("--out", default="universe_candidate_v4_japan.csv")
    ap.add_argument("--expected-count", type=int, default=225)
    args = ap.parse_args()

    out_path = Path(args.out)

    # Never leave a stale/empty file looking like a valid build.
    if out_path.exists():
        out_path.unlink()

    r = requests.get(
        OFFICIAL_URL,
        headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
        timeout=60,
    )
    r.raise_for_status()
    html = r.text

    tables = pd.read_html(io.StringIO(html))
    rows = []

    for t in tables:
        t = flatten_columns(t)
        code_col = find_exactish(t.columns, "code")
        company_col = find_exactish(t.columns, "company name")
        if code_col is None or company_col is None:
            continue

        for _, rr in t.iterrows():
            code = normalized_name(rr[code_col]).upper()
            company = normalized_name(rr[company_col])

            code = re.sub(r"\.0$", "", code)
            code = re.sub(r"\[[^\]]+\]", "", code).strip()

            if not CODE_RE.fullmatch(code):
                continue
            if not company or company.lower() in {"nan", "company name"}:
                continue

            rows.append(
                {
                    "ticker": f"{code}.T",
                    "asset_class": "equity",
                    "source_index": "nikkei225",
                    "sector": "UNKNOWN",
                    "company": company,
                    "security_code": code,
                    "source_url": OFFICIAL_URL,
                }
            )

    if not rows:
        raise SystemExit(
            "V4_CANDIDATE_BUILD: FAIL no constituent rows parsed from official Nikkei page"
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["security_code"], keep="first")
    out = out.sort_values("security_code").reset_index(drop=True)

    expected = int(args.expected_count)
    if len(out) != expected:
        preview = out[["security_code", "company"]].head(10).to_dict(orient="records")
        raise SystemExit(
            f"V4_CANDIDATE_BUILD: FAIL parsed {len(out)} unique constituents, "
            f"expected exactly {expected}. No CSV was written. Preview={preview}"
        )

    if out["ticker"].duplicated().any():
        raise SystemExit("V4_CANDIDATE_BUILD: FAIL duplicate Yahoo tickers")

    if not out["ticker"].str.endswith(".T").all():
        raise SystemExit("V4_CANDIDATE_BUILD: FAIL malformed Tokyo Yahoo ticker")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Re-read to ensure a genuinely valid non-empty CSV was written.
    check = pd.read_csv(out_path)
    if len(check) != expected or "ticker" not in check.columns:
        out_path.unlink(missing_ok=True)
        raise SystemExit("V4_CANDIDATE_BUILD: FAIL post-write verification failed")

    print(f"official source: {OFFICIAL_URL}")
    print(f"tables parsed: {len(tables)}")
    print(f"unique constituents: {len(out)}")
    print(f"first ticker: {out.iloc[0]['ticker']}")
    print(f"last ticker: {out.iloc[-1]['ticker']}")
    print(f"output: {out_path}")
    print("V4_CANDIDATE_BUILD: PASS")


if __name__ == "__main__":
    main()

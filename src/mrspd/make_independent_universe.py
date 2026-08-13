from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _yf_symbol(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove all discovery tickers from a candidate universe to create a genuinely disjoint validation universe"
    )
    p.add_argument("--candidate-universe", required=True)
    p.add_argument("--discovery-panel", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-tickers", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cand = pd.read_csv(Path(args.candidate_universe))
    if "ticker" not in cand.columns or "asset_class" not in cand.columns:
        raise ValueError("candidate universe must contain ticker,asset_class columns")
    discovery = pd.read_csv(Path(args.discovery_panel), usecols=["ticker"])
    used = {_yf_symbol(x) for x in discovery["ticker"].dropna().unique()}
    cand = cand.copy()
    cand["ticker"] = cand["ticker"].map(_yf_symbol)
    cand = cand.drop_duplicates(subset=["ticker"], keep="first")
    before = len(cand)
    out = cand[~cand["ticker"].isin(used)].copy()
    # SPY is automatically downloaded by mrspd.py for residualization and need
    # not be a validation-panel asset. Removing it avoids accidental overlap.
    out = out[out["ticker"] != "SPY"].copy()
    if len(out) < int(args.min_tickers):
        raise ValueError(
            f"Only {len(out)} disjoint tickers remain (< min-tickers={args.min_tickers}). "
            "Supply a broader candidate universe that was not used in discovery."
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out), index=False)
    print(f"candidate tickers: {before}")
    print(f"discovery tickers excluded: {len(used)}")
    print(f"independent tickers written: {len(out)}")
    print(f"INDEPENDENT_UNIVERSE: PASS -> {args.out}")


if __name__ == "__main__":
    main()

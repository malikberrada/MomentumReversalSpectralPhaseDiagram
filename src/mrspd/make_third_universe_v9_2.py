from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd

CHUNK=250_000

def norm(x: object) -> str:
    return str(x).strip().upper()

def panel_tickers(path: Path) -> set[str]:
    out=set()
    for c in pd.read_csv(path,usecols=['ticker'],chunksize=CHUNK):
        out.update(norm(x) for x in c['ticker'].dropna().unique())
    return out

def main():
    ap=argparse.ArgumentParser(description='Create V3 universe disjoint from D and V2 before price inspection')
    ap.add_argument('--candidate-universe',required=True)
    ap.add_argument('--discovery-panel',required=True)
    ap.add_argument('--second-panel',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--min-tickers',type=int,default=100)
    a=ap.parse_args()
    c=pd.read_csv(a.candidate_universe)
    if not {'ticker','asset_class'}.issubset(c.columns):
        raise ValueError('candidate universe must contain ticker,asset_class')
    c=c.copy(); c['ticker']=c['ticker'].map(norm); c=c.drop_duplicates('ticker')
    used=panel_tickers(Path(a.discovery_panel))|panel_tickers(Path(a.second_panel))
    o=c[~c['ticker'].isin(used)].copy().sort_values('ticker').reset_index(drop=True)
    if len(o)<a.min_tickers:
        raise ValueError(f'only {len(o)} V3 tickers remain after excluding previously observed tickers')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); o.to_csv(a.out,index=False)
    print(f'THIRD_UNIVERSE_V9_2: PASS candidates={len(c)} overlap_removed={len(c)-len(o)} written={len(o)} -> {a.out}')
if __name__=='__main__': main()


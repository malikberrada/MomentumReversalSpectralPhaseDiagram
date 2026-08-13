from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .transport_v9 import normalize_ticker, panel_metadata_streaming

def main():
 p=argparse.ArgumentParser(description='Create a third MRSPD universe disjoint from both previously observed universes')
 p.add_argument('--candidate-universe',required=True); p.add_argument('--discovery-panel',required=True); p.add_argument('--second-panel',required=True); p.add_argument('--out',required=True); p.add_argument('--min-tickers',type=int,default=100)
 a=p.parse_args(); c=pd.read_csv(a.candidate_universe)
 if not {'ticker','asset_class'}.issubset(c.columns): raise ValueError('candidate universe must contain ticker,asset_class')
 c=c.copy(); c['ticker']=c['ticker'].map(normalize_ticker); c=c.drop_duplicates('ticker')
 d=panel_metadata_streaming(Path(a.discovery_panel)); s=panel_metadata_streaming(Path(a.second_panel)); used=set(d['tickers'])|set(s['tickers'])
 out=c[~c['ticker'].isin(used)].copy()
 if len(out)<a.min_tickers: raise ValueError(f'only {len(out)} third-universe tickers remain after excluding {len(used)} previously observed tickers')
 out.to_csv(a.out,index=False); print(f'THIRD_UNIVERSE: PASS candidates={len(c)} excluded_used={len(c)-len(out)} written={len(out)} -> {a.out}')
if __name__=='__main__': main()

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


def norm(x: object) -> str:
    return str(x).strip().upper()

def canonical_hash(doc: dict) -> str:
    x=dict(doc); x.pop('protocol_sha256',None)
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def file_sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description='Bind frozen v9.2 protocol to V3 ticker list before building/downloading V3 panel')
    ap.add_argument('--protocol',required=True); ap.add_argument('--third-universe',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--min-panel-coverage',type=float,default=None)
    a=ap.parse_args()
    p=json.loads(Path(a.protocol).read_text(encoding='utf-8'))
    if p.get('protocol_sha256')!=canonical_hash(p): raise ValueError('input protocol SHA256 mismatch')
    if p.get('status')!='FROZEN_BEFORE_THIRD_UNIVERSE_PRICE_INSPECTION': raise ValueError('protocol is not in pre-V3 frozen state')
    u=pd.read_csv(a.third_universe)
    if 'ticker' not in u.columns: raise ValueError('third universe must contain ticker')
    tickers=sorted(set(norm(x) for x in u['ticker'].dropna()))
    if not tickers: raise ValueError('empty third universe')
    tsha=hashlib.sha256('\n'.join(tickers).encode()).hexdigest()
    requested = p['third_universe_pass_rule']['minimum_bound_ticker_coverage_fraction'] if a.min_panel_coverage is None else float(a.min_panel_coverage)
    frozen_min=float(p['third_universe_pass_rule']['minimum_bound_ticker_coverage_fraction'])
    if abs(requested-frozen_min)>1e-12: raise ValueError(f'min coverage is frozen at {frozen_min}; do not override it')
    p['third_universe_design']={
        'ticker_count':len(tickers),
        'tickers':tickers,
        'ticker_set_sha256':tsha,
        'universe_file_sha256':file_sha256(Path(a.third_universe)),
        'min_panel_coverage_fraction':frozen_min,
        'binding_status':'BOUND_BEFORE_V3_PANEL_CONSTRUCTION',
    }
    p['status']='FROZEN_AND_BOUND_BEFORE_THIRD_UNIVERSE_PANEL_CONSTRUCTION'
    p['parent_protocol_sha256']=p['protocol_sha256']
    p.pop('protocol_sha256',None); p['protocol_sha256']=canonical_hash(p)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(p,indent=2),encoding='utf-8')
    print(json.dumps({'status':'BOUND_V9_2','third_ticker_count':len(tickers),'ticker_set_sha256':tsha,'protocol_sha256':p['protocol_sha256'],'out':str(out)},indent=2))
if __name__=='__main__': main()


from __future__ import annotations
import argparse,json
from pathlib import Path
from .transport_v9 import bind_third_universe_protocol

def main():
 p=argparse.ArgumentParser(description='Bind a frozen MRSPD v9 hypothesis protocol to the untouched third-universe ticker list before panel construction')
 p.add_argument('--protocol',required=True); p.add_argument('--third-universe',required=True); p.add_argument('--out',required=True); p.add_argument('--min-panel-coverage',type=float,default=.80)
 a=p.parse_args(); q=bind_third_universe_protocol(protocol_path=Path(a.protocol),third_universe_path=Path(a.third_universe),out_path=Path(a.out),min_panel_coverage_fraction=a.min_panel_coverage); print(json.dumps(q,indent=2,default=str))
if __name__=='__main__': main()

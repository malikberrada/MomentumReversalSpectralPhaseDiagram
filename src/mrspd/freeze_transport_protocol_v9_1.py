from __future__ import annotations
import argparse,json
from pathlib import Path
from .transport_v9_1 import freeze_protocol_v91

def main():
    p=argparse.ArgumentParser(); p.add_argument('--exploratory-summary',required=True); p.add_argument('--selected-hypothesis',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    d=freeze_protocol_v91(exploratory_summary=Path(a.exploratory_summary),selected_hypothesis=Path(a.selected_hypothesis),out=Path(a.out)); print(json.dumps(d,indent=2))
if __name__=='__main__': main()

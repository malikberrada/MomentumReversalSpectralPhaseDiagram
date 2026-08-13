from __future__ import annotations
import argparse,json
from pathlib import Path
from .transport_v9 import freeze_transport_protocol

def main():
 p=argparse.ArgumentParser(description='Freeze MRSPD v9 third-universe transport protocol')
 p.add_argument('--exploratory-summary',required=True); p.add_argument('--selected-hypothesis',required=True)
 p.add_argument('--discovery-panel',required=True); p.add_argument('--second-panel',required=True); p.add_argument('--out',required=True)
 a=p.parse_args(); doc=freeze_transport_protocol(exploratory_summary_path=Path(a.exploratory_summary),selected_hypothesis_path=Path(a.selected_hypothesis),discovery_panel_path=Path(a.discovery_panel),second_panel_path=Path(a.second_panel),out_path=Path(a.out)); print(json.dumps(doc,indent=2,default=str))
if __name__=='__main__': main()

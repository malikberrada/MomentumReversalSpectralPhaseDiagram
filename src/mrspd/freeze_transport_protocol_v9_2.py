from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .robust_transport_v9_2 import PROTOCOL_VERSION, canonical_json_sha256

def file_sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exploratory-summary',required=True)
    ap.add_argument('--selected-hypothesis',required=True)
    ap.add_argument('--discovery-panel',required=True)
    ap.add_argument('--second-panel',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    summary=json.loads(Path(a.exploratory_summary).read_text(encoding='utf-8'))
    sel=json.loads(Path(a.selected_hypothesis).read_text(encoding='utf-8'))
    if not sel.get('selected'):
        raise SystemExit('REFUSE_TO_FREEZE: v9.2 has no transport candidate identified in both development universes and both modes.')
    r=summary['rules']
    protocol={
      'protocol_version':PROTOCOL_VERSION,
      'status':'FROZEN_BEFORE_THIRD_UNIVERSE_ANALYSIS',
      'development_only_sources':{
        'discovery_panel_sha256':file_sha256(Path(a.discovery_panel)),
        'second_panel_sha256':file_sha256(Path(a.second_panel)),
      },
      'hypothesis':{
        'coordinate':'global empirical q_psi percentile within universe x date x return_mode x horizon',
        'response':'cross-sectional Huber-clipped phase_product using median/MAD and fixed c=1.345',
        'tail_direction':'upper','q0':sel['q0'],'tail_bins':2,
        'claim':'persistent OOS negative robust phase response in both upper-tail sub-bands for raw and market_residual',
      },
      'expected_development_hc':{'discovery':sel['discovery_hc_grid'],'second':sel['second_hc_grid']},
      'horizons':summary['horizons'],
      'rules':r,
      'third_universe_pass_rule':{
        'independence_required':True,'both_return_modes_required':True,
        'persistent_tail_required':True,
        'localization_envelope':[min(sel['discovery_hc_grid'],sel['second_hc_grid']),max(sel['discovery_hc_grid'],sel['second_hc_grid'])],
        'localization_tolerance_days':21,
      },
      'amendment_note':'This is a new hypothesis developed after v8/v9 nonreplication. It does not amend or relabel earlier confirmatory results.'
    }
    protocol['protocol_sha256']=canonical_json_sha256(protocol)
    Path(a.out).write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps({'status':'FROZEN','protocol_sha256':protocol['protocol_sha256'],'out':a.out},indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import argparse,json
from pathlib import Path
from .transport_v9 import add_cross_sectional_percentile, audit_bound_third_panel, canonical_json_sha256, evaluate_third_universe, panel_metadata_streaming, read_panel_streaming

def main():
 p=argparse.ArgumentParser(description='Locked MRSPD v9 validation on an untouched third universe')
 p.add_argument('--protocol',required=True); p.add_argument('--discovery-panel',required=True); p.add_argument('--second-panel',required=True); p.add_argument('--third-panel',required=True); p.add_argument('--out',required=True)
 a=p.parse_args(); protocol=json.loads(Path(a.protocol).read_text(encoding='utf-8'))
 if protocol.get('protocol_sha256')!=canonical_json_sha256(protocol): raise ValueError('protocol SHA256 mismatch')
 dm=panel_metadata_streaming(Path(a.discovery_panel)); sm=panel_metadata_streaming(Path(a.second_panel)); tm=panel_metadata_streaming(Path(a.third_panel))
 used=set(dm['tickers'])|set(sm['tickers']); overlap=sorted(used&set(tm['tickers']))
 independence={'pass':not overlap,'third_ticker_count':tm['ticker_count'],'overlap_count':len(overlap),'overlap_preview':overlap[:20]}
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'third_universe_independence_audit.json').write_text(json.dumps(independence,indent=2),encoding='utf-8')
 if overlap: raise ValueError(f'third universe is not independent: {overlap[:20]}')
 third=read_panel_streaming(Path(a.third_panel),label='third_universe'); third=third[third['horizon'].isin(protocol['coarse_horizons'])].reset_index(drop=True); add_cross_sectional_percentile(third)
 folds,regs,hc,cons=evaluate_third_universe(third_panel=third,protocol=protocol)
 folds.to_csv(out/'third_tail_fold_certification.csv',index=False); regs.to_csv(out/'third_tail_regimes.csv',index=False); hc.to_csv(out/'third_tail_hc.csv',index=False)
 env=protocol['hypothesis']['expected_hc_envelope']; tol=int(protocol['hypothesis']['localization_tolerance_days']); vh=cons.get('hc_grid')
 phenomenon=bool(cons.get('identified')); localization=bool(phenomenon and vh is not None and int(env[0])-tol <= int(vh) <= int(env[1])+tol)
 verdict={'independence_pass':True,'phenomenon_confirmed':phenomenon,'localization_confirmed':localization,'validation_hc_grid':int(vh) if vh is not None else None,
          'frozen_expected_hc_envelope':env,'localization_tolerance_days':tol,'overall_confirmatory_pass':bool(phenomenon and localization),
          'interpretation':'CONFIRMED' if phenomenon and localization else 'NOT_CONFIRMED','consensus':cons,'protocol_sha256':protocol['protocol_sha256']}
 (out/'third_universe_validation_summary.json').write_text(json.dumps(verdict,indent=2,default=str),encoding='utf-8'); print(json.dumps(verdict,indent=2,default=str))
if __name__=='__main__': main()

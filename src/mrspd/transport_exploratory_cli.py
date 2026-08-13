from __future__ import annotations
import argparse, json
from pathlib import Path
from .transport_v9 import DEFAULT_COARSE_HORIZONS, run_transport_exploratory

def _hs(s:str): return tuple(sorted(set(int(x) for x in s.split(',') if x.strip())))
def main():
    p=argparse.ArgumentParser(description='MRSPD v9 exploratory percentile-transport analysis on discovery + second universe')
    p.add_argument('--discovery-panel',required=True); p.add_argument('--second-panel',required=True); p.add_argument('--out',required=True)
    p.add_argument('--discovery-metadata'); p.add_argument('--second-metadata')
    p.add_argument('--horizons',type=_hs,default=DEFAULT_COARSE_HORIZONS)
    p.add_argument('--bootstrap-reps',type=int,default=1000); p.add_argument('--max-fit-rows',type=int,default=250000)
    p.add_argument('--alpha',type=float,default=.05); p.add_argument('--min-effect-sigma',type=float,default=.02)
    p.add_argument('--min-folds',type=int,default=3); p.add_argument('--min-fold-fraction',type=float,default=.75)
    p.add_argument('--walkforward-splits',type=int,default=4); p.add_argument('--min-train-frac',type=float,default=.5)
    p.add_argument('--stride',type=int,default=5); p.add_argument('--seed',type=int,default=20260812)
    a=p.parse_args()
    s=run_transport_exploratory(discovery_panel_path=Path(a.discovery_panel),validation_panel_path=Path(a.second_panel),out_dir=Path(a.out),
        discovery_metadata_path=Path(a.discovery_metadata) if a.discovery_metadata else None,
        validation_metadata_path=Path(a.second_metadata) if a.second_metadata else None,
        horizons=a.horizons,bootstrap_reps=a.bootstrap_reps,max_fit_rows=a.max_fit_rows,alpha=a.alpha,min_effect_sigma=a.min_effect_sigma,
        min_folds=a.min_folds,min_fold_fraction=a.min_fold_fraction,splits=a.walkforward_splits,min_train_frac=a.min_train_frac,stride=a.stride,seed=a.seed)
    print(json.dumps(s,indent=2,default=str))
if __name__=='__main__': main()

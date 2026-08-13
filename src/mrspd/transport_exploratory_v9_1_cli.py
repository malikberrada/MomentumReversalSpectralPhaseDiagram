from __future__ import annotations
import argparse,json
from pathlib import Path
from .transport_v9_1 import DEFAULT_Q0_GRID, run_exploratory_v91

def _hs(s): return tuple(sorted(set(int(x) for x in s.split(',') if x.strip())))
def _qs(s): return tuple(float(x) for x in s.split(',') if x.strip())
def main():
    p=argparse.ArgumentParser(description='MRSPD v9.1 scale-aligned percentile transport development analysis')
    p.add_argument('--discovery-panel',required=True); p.add_argument('--second-panel',required=True); p.add_argument('--out',required=True)
    p.add_argument('--discovery-metadata'); p.add_argument('--second-metadata'); p.add_argument('--horizons',type=_hs,default=(63,72,84,96,105,112,119,126,133,140,147,168)); p.add_argument('--q0-grid',type=_qs,default=DEFAULT_Q0_GRID)
    p.add_argument('--bootstrap-reps',type=int,default=1000); p.add_argument('--alpha',type=float,default=.05); p.add_argument('--min-effect-sigma',type=float,default=.02); p.add_argument('--min-folds',type=int,default=3); p.add_argument('--min-fold-fraction',type=float,default=.75); p.add_argument('--walkforward-splits',type=int,default=4); p.add_argument('--min-train-frac',type=float,default=.5); p.add_argument('--stride',type=int,default=5); p.add_argument('--seed',type=int,default=20260812)
    a=p.parse_args(); s=run_exploratory_v91(discovery_panel_path=Path(a.discovery_panel),second_panel_path=Path(a.second_panel),out_dir=Path(a.out),discovery_metadata_path=Path(a.discovery_metadata) if a.discovery_metadata else None,second_metadata_path=Path(a.second_metadata) if a.second_metadata else None,horizons=a.horizons,q0_grid=a.q0_grid,bootstrap_reps=a.bootstrap_reps,alpha=a.alpha,min_effect_sigma=a.min_effect_sigma,min_folds=a.min_folds,min_fold_fraction=a.min_fold_fraction,splits=a.walkforward_splits,min_train_frac=a.min_train_frac,stride=a.stride,seed=a.seed); print(json.dumps(s,indent=2,default=str))
if __name__=='__main__': main()

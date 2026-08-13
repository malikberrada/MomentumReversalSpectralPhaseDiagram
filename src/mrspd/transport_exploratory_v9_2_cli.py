from __future__ import annotations
import argparse, json
from pathlib import Path
from .robust_transport_v9_2 import run_exploratory

def csv_ints(s): return [int(x.strip()) for x in s.split(',') if x.strip()]
def csv_floats(s): return [float(x.strip()) for x in s.split(',') if x.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--discovery-panel',required=True)
    p.add_argument('--second-panel',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--horizons',default='63,72,84,96,105,112,119,126,133,140,147,168,189,210,231,252')
    p.add_argument('--q0-grid',default='0.60,0.65,0.70,0.75,0.80,0.85,0.90')
    p.add_argument('--bootstrap-reps',type=int,default=1000)
    p.add_argument('--alpha',type=float,default=0.05)
    p.add_argument('--min-effect-sigma',type=float,default=0.02)
    p.add_argument('--min-folds',type=int,default=3)
    p.add_argument('--min-fold-fraction',type=float,default=0.75)
    p.add_argument('--hc-min-consecutive',type=int,default=3)
    p.add_argument('--hc-min-tail-fraction',type=float,default=0.8)
    p.add_argument('--walkforward-splits',type=int,default=4)
    p.add_argument('--min-train-frac',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=20260812)
    a=p.parse_args()
    result=run_exploratory(Path(a.discovery_panel),Path(a.second_panel),Path(a.out),csv_ints(a.horizons),csv_floats(a.q0_grid),
        bootstrap_reps=a.bootstrap_reps,alpha=a.alpha,min_effect_sigma=a.min_effect_sigma,min_folds=a.min_folds,
        min_fold_fraction=a.min_fold_fraction,hc_min_consecutive=a.hc_min_consecutive,hc_min_tail_fraction=a.hc_min_tail_fraction,
        walkforward_splits=a.walkforward_splits,min_train_frac=a.min_train_frac,seed=a.seed)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()

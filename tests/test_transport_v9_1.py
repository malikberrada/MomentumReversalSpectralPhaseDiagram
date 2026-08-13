import json
from types import SimpleNamespace
import numpy as np, pandas as pd
from mrspd.transport_v9_1 import V91Config,_aligned_tail_fold_certification,add_normalized_coordinates,select_hypothesis

def _panel():
    dates=pd.bdate_range('2020-01-01',periods=80); rows=[]
    for d in dates:
        for i in range(40): rows.append({'date':d,'ticker':f'T{i}','return_mode':'raw','horizon':126,'psi_primary':i,'phase_product':-0.5})
    return pd.DataFrame(rows)

def test_normalized_coordinate():
    p=_panel(); add_normalized_coordinates(p); assert p.q_psi_global.between(0,1).all()

def test_aligned_effect_scale_is_date_level(monkeypatch):
    p=_panel(); add_normalized_coordinates(p); tr=p.iloc[:2400].copy(); te=p.iloc[2400:].copy()
    # Inject huge row-level dispersion outside the tested upper tail; date-level upper-tail means remain stable.
    tr.loc[tr.q_psi_global<.5,'phase_product']=[1000 if i%2 else -1000 for i in range((tr.q_psi_global<.5).sum())]
    cfg=V91Config(bootstrap_reps=50,min_obs_per_tail_bin=20,min_dates_per_tail_bin=10,min_train_dates_per_tail_bin=10)
    r=_aligned_tail_fold_certification(tr,te,qcol='q_psi_global',q0=.75,cfg=cfg,block_len=2,seed=1)
    assert r['eligible']; details=json.loads(r['tail_bins_json']); assert max(x['epsilon_bin'] for x in details)<0.1

def test_selection_prefers_transportable():
    rec=[{'normalization':'global','q0':.75,'min_effect_sigma':.02,'discovery_consensus':{'identified':True,'hc_grid':133},'second_consensus':{'identified':True,'hc_grid':140},'discovery_min_post_onset_support':1.0,'second_min_post_onset_support':.75}]
    s=select_hypothesis(rec); assert s['selected'] and s['q0']==.75

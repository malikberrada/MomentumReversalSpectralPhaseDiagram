from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

from mrspd import transport_v9 as tv


def _panel(tickers=('A','B','C','D','E','F','G','H'), start='2020-01-01', ndates=180, horizons=(63,84,105,126), onset=105):
    dates=pd.bdate_range(start,periods=ndates)
    rows=[]
    for mode in ['raw','market_residual']:
        for h in horizons:
            for di,d in enumerate(dates):
                for ti,t in enumerate(tickers):
                    psi=float(ti)+0.01*np.sin(di/7)
                    # Strong negative upper quartile only after onset; elsewhere mixed/positive.
                    rank=(ti+1)/len(tickers)
                    y=(-1.0-0.05*np.cos(di/5)) if (h>=onset and rank>=0.75) else (0.15*np.sin(di/9)+0.10)
                    rows.append((d,t,'us_stock',mode,h,psi,y))
    return pd.DataFrame(rows,columns=['date','ticker','asset_class','return_mode','horizon','psi_primary','phase_product'])


def test_percentile_is_intra_date_mode_horizon():
    p=_panel(ndates=3,horizons=(63,))
    tv.add_cross_sectional_percentile(p)
    g=p[(p.return_mode=='raw')&(p.horizon==63)&(p.date==p.date.min())].sort_values('psi_primary')
    assert np.allclose(g.q_psi.to_numpy(),np.arange(1,9)/8)


def test_tail_edges_quartiles():
    assert np.allclose(tv._tail_edges('lower',.25,2),[0,.125,.25])
    assert np.allclose(tv._tail_edges('upper',.25,2),[.75,.875,1.0])


def test_tail_certification_does_not_require_root_free():
    p=_panel(ndates=220,horizons=(105,))
    tv.add_cross_sectional_percentile(p)
    tr=p[p.date < p.date.sort_values().unique()[140]]
    te=p[p.date >= p.date.sort_values().unique()[140]]
    cfg=tv.TransportConfig(bootstrap_reps=999,min_obs_per_tail_bin=20,min_dates_per_tail_bin=20,alpha=.05,min_effect_sigma=.02)
    out=tv._tail_fold_certification(tr[(tr.return_mode=='raw')],te[(te.return_mode=='raw')],direction='upper',tcfg=cfg,block_len=3,seed=1)
    assert out['eligible']
    assert out['tail_negative']


def test_selector_requires_both_universes_and_both_modes_consensus():
    c=[
      {'tail_direction':'lower','tail_width':.25,'tail_bins':2,'discovery_consensus':{'identified':False},'validation_consensus':{'identified':False}},
      {'tail_direction':'upper','tail_width':.25,'tail_bins':2,'discovery_consensus':{'identified':True,'hc_grid':126},'validation_consensus':{'identified':True,'hc_grid':133}},
    ]
    s=tv.select_transport_hypothesis(c,[63,84,105,126,133,147])
    assert s['selected'] and s['tail_direction']=='upper'
    assert s['expected_hc_envelope']==[126,133]


def test_freeze_protocol_refuses_unselected(tmp_path:Path):
    summary={'rules':{},'coarse_horizons':[63,84,105]}
    sel={'selected':False}
    (tmp_path/'sum.json').write_text(json.dumps(summary)); (tmp_path/'sel.json').write_text(json.dumps(sel))
    try:
        tv.freeze_transport_protocol(exploratory_summary_path=tmp_path/'sum.json',selected_hypothesis_path=tmp_path/'sel.json',discovery_panel_path=tmp_path/'d.csv.gz',second_panel_path=tmp_path/'s.csv.gz',out_path=tmp_path/'p.json')
    except ValueError as e:
        assert 'No transportable' in str(e)
    else:
        raise AssertionError('expected refusal')


def test_panel_metadata_and_third_protocol_hash(tmp_path:Path):
    d=_panel(tickers=('A','B','C','D'),ndates=5,horizons=(63,84,105)); s=_panel(tickers=('E','F','G','H'),start='2021-01-01',ndates=5,horizons=(63,84,105))
    dp=tmp_path/'d.csv.gz'; sp=tmp_path/'s.csv.gz'; d.to_csv(dp,index=False,compression='gzip'); s.to_csv(sp,index=False,compression='gzip')
    summary={'coarse_horizons':[63,84,105],'rules':{'spline_knots':7,'bootstrap_reps':1000,'max_fit_rows':250000,'alpha':.05,'min_effect_sigma':.02,'tail_width':.25,'tail_bins':2,'min_folds':3,'min_fold_fraction':.75,'hc_min_consecutive':3,'hc_min_tail_fraction':.8,'walkforward_splits':4,'min_train_frac':.5,'stride':5,'seed':20260812}}
    sel={'selected':True,'tail_direction':'upper','tail_width':.25,'tail_bins':2,'expected_hc_envelope':[84,105],'localization_tolerance_days':21}
    (tmp_path/'sum.json').write_text(json.dumps(summary)); (tmp_path/'sel.json').write_text(json.dumps(sel))
    p=tv.freeze_transport_protocol(exploratory_summary_path=tmp_path/'sum.json',selected_hypothesis_path=tmp_path/'sel.json',discovery_panel_path=dp,second_panel_path=sp,out_path=tmp_path/'protocol.json')
    assert p['protocol_version']==tv.PROTOCOL_VERSION
    assert p['protocol_sha256']==tv.canonical_json_sha256(p)
    assert set(p['provenance']['discovery_panel_metadata']['tickers']).isdisjoint(set(p['provenance']['second_universe_panel_metadata']['tickers']))

def test_bind_third_universe_and_panel_audit(tmp_path:Path):
    base={'protocol_version':tv.PROTOCOL_VERSION,'status':'FROZEN_BEFORE_THIRD_UNIVERSE_VALIDATION','hypothesis':{},'provenance':{},'coarse_horizons':[63,84,105],'rules':{},'primary_endpoints':{},'guardrails':[]}
    base['protocol_sha256']=tv.canonical_json_sha256(base)
    bp=tmp_path/'base.json'; bp.write_text(json.dumps(base))
    u=pd.DataFrame({'ticker':['X','Y','Z']+[f'T{i}' for i in range(60)],'asset_class':['intl']*63})
    up=tmp_path/'u.csv'; u.to_csv(up,index=False)
    q=tv.bind_third_universe_protocol(protocol_path=bp,third_universe_path=up,out_path=tmp_path/'bound.json',min_panel_coverage_fraction=.8)
    assert q['status']=='FROZEN_WITH_THIRD_UNIVERSE_BEFORE_PANEL_ANALYSIS'
    meta={'tickers':q['third_universe_design']['tickers'][:55]}
    a=tv.audit_bound_third_panel(q,meta)
    assert a['pass']
    meta2={'tickers':q['third_universe_design']['tickers'][:30]+['NOT_ALLOWED']}
    assert not tv.audit_bound_third_panel(q,meta2)['pass']

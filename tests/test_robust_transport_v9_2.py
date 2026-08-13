import numpy as np, pandas as pd
from mrspd.robust_transport_v9_2 import add_global_percentile_and_robust_response, expanding_walkforward_dates, _bh_holm

def test_huber_clip_kills_single_extreme():
    n=100
    d=pd.DataFrame({'date':[pd.Timestamp('2020-01-01')]*n,'ticker':[f'T{i}' for i in range(n)],
        'return_mode':['raw']*n,'horizon':[168]*n,'psi_primary':np.arange(n,dtype=float),
        'phase_product':np.r_[np.linspace(-1,1,n-1),1000.0]})
    z=add_global_percentile_and_robust_response(d)
    assert z.phase_product_robust.max() < 1000
    assert z.was_clipped.sum() >= 1
    assert z.q_psi.min()>0 and z.q_psi.max()==1

def test_walkforward_is_forward_only():
    ds=pd.date_range('2020-01-01',periods=100,freq='D')
    fs=expanding_walkforward_dates(ds,4,.5)
    assert len(fs)==4
    for tr,te in fs:
        assert tr.max() < te.min()

def test_holm():
    p=[.001,.01,.2]
    h=_bh_holm(p,.05)
    assert h[:2]==[True,True] and h[2] is False

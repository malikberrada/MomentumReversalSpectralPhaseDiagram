import numpy as np
import pandas as pd

from mrspd import pipeline as mrspd


def ar1(phi: float, n: int = 6000, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n))


def test_sepp_lucic_score_sign_tracks_ar1():
    cfg = mrspd.Config(spectral_window=1000, spectral_spans=(21,), horizons=(21,))
    pos = mrspd._spectral_bank(ar1(+0.15), cfg)["psi_21"].dropna().tail(1000).median()
    neg = mrspd._spectral_bank(ar1(-0.15), cfg)["psi_21"].dropna().tail(1000).median()
    assert pos > 0
    assert neg < 0


def test_future_sum_alignment():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    f = mrspd._future_sum(x, 2)
    assert f.iloc[1] == 3.0 + 4.0
    assert f.iloc[2] == 4.0 + 5.0

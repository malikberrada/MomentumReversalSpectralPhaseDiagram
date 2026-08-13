from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

native = pytest.importorskip("mrspd_native")


def pandas_ref(z: np.ndarray, spans=(5, 21), window=200, minp=120, horizons=(5, 21)):
    s = pd.Series(z)
    var_z = s.rolling(window, min_periods=minp).var(ddof=0)
    psis = []
    for span in spans:
        nu = (span - 1.0) / (span + 1.0)
        lam = s.ewm(alpha=1.0 - nu, adjust=False, min_periods=span).mean().shift(1)
        cov = s.rolling(window, min_periods=minp).cov(lam)
        psis.append((nu / (1.0 - nu)) * (cov / var_z.replace(0.0, np.nan)).to_numpy())
    past, future = [], []
    for h in horizons:
        past.append(s.rolling(h, min_periods=h).sum().to_numpy())
        future.append(s.shift(-h).rolling(h, min_periods=h).sum().to_numpy())
    return np.asarray(psis), np.asarray(past), np.asarray(future)


def test_native_cpu_matches_reference_contiguous_series():
    rng = np.random.default_rng(123)
    z = rng.normal(size=1200)
    z[:40] = np.nan
    spans = np.array([5, 21], np.int32)
    horizons = np.array([5, 21], np.int32)
    psi, past, future = native.feature_bank_batch(z[None, :], spans, 200, 120, horizons, "cpu")
    rpsi, rpast, rfuture = pandas_ref(z)
    mask = np.isfinite(rpsi) & np.isfinite(psi[0])
    assert np.nanmax(np.abs(psi[0][mask] - rpsi[mask])) < 1e-10
    assert np.allclose(past[0], rpast, equal_nan=True, atol=1e-12)
    assert np.allclose(future[0], rfuture, equal_nan=True, atol=1e-12)

import numpy as np
import pandas as pd

from mrspd.critical_horizon import (
    _aggregate_horizon_regimes,
    _apply_horizon_family_holm,
    _find_onset,
    _holm_adjust,
)


def test_holm_adjust_known_values():
    p = np.array([0.01, 0.04, 0.03])
    got = _holm_adjust(p)
    assert np.allclose(got, [0.03, 0.06, 0.06])


def test_find_onset_brackets_first_persistent_r_tail():
    h = [63, 72, 84, 96, 105, 112, 126]
    r = [False, False, True, True, True, True, True]
    got = _find_onset(h, r, min_consecutive=3, min_tail_fraction=0.8)
    assert got["identified"] is True
    assert got["hc_lower_exclusive"] == 72
    assert got["hc_upper_inclusive"] == 84
    assert got["strict_r_tail"] is True


def test_find_onset_rejects_short_or_unstable_run():
    h = [63, 72, 84, 96, 105, 112, 126]
    r = [False, False, True, True, False, True, False]
    got = _find_onset(h, r, min_consecutive=3, min_tail_fraction=0.8)
    assert got["identified"] is False


def test_find_onset_rejects_later_m_certification():
    h = [63, 72, 84, 96, 105, 112]
    r = [False, True, True, True, True, True]
    m = [False, False, False, False, False, True]
    got = _find_onset(h, r, m, min_consecutive=3, min_tail_fraction=0.8)
    assert got["identified"] is False


def test_horizon_holm_and_cross_fold_aggregation():
    rows = []
    for fold in range(1, 5):
        for h, p in [(63, 0.2), (84, 0.001), (96, 0.001), (112, 0.001)]:
            rows.append({
                "return_mode": "raw",
                "fold": fold,
                "horizon": h,
                "eligible": True,
                "phase": "R" if h >= 84 else "",
                "candidate_p_iut": p if h >= 84 else np.nan,
                "train_n_roots": 0,
            })
    z = _apply_horizon_family_holm(pd.DataFrame(rows), 0.05)
    assert int(z["certified_r_only_fold"].sum()) == 12
    agg = _aggregate_horizon_regimes(z, min_folds=3, min_fold_fraction=0.75)
    q = agg.set_index("horizon")
    assert q.loc[84, "certified_regime"] == "R"
    assert bool(q.loc[84, "certified_r_only"])
    assert q.loc[63, "certified_regime"] == "UNRESOLVED"

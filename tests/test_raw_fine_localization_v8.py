import pandas as pd

from mrspd.critical_horizon_refine import (
    _aggregate_fixed_sequence_regimes,
    _coarse_fit_subset,
    _fixed_sequence_gatekeeping,
    _refined_onset,
    _v6_fit_seed,
)


def _fine_rows(mode="raw", folds=4):
    rows = []
    for fold in range(1, folds + 1):
        for h in range(120, 134):
            rows.append({
                "return_mode": mode,
                "fold": fold,
                "horizon": h,
                "eligible": True,
                "certified_single_phase": True,
                "phase": "R",
                "candidate_p_iut": 0.01,
                "train_n_roots": 0,
            })
    return pd.DataFrame(rows)


def _coarse_fold_anchor(mode="raw", anchor=133, pass_folds=(1, 2, 3)):
    return pd.DataFrame([
        {
            "return_mode": mode,
            "fold": fold,
            "horizon": anchor,
            "certified_r_only_fold": fold in pass_folds,
        }
        for fold in range(1, 5)
    ])


def _coarse_regimes(mode="raw"):
    return pd.DataFrame({
        "return_mode": [mode] * 4,
        "horizon": [133, 140, 147, 168],
        "certified_regime": ["R", "R", "R", "R"],
    })


def test_dense_measurement_rows_cannot_enter_surface_fit():
    d = pd.DataFrame({"horizon": [63, 120, 126, 133, 140, 147], "x": range(6)})
    out = _coarse_fit_subset(d, [63, 126, 133, 140, 147])
    assert out["horizon"].tolist() == [63, 126, 133, 140, 147]
    assert 120 not in set(out["horizon"])


def test_refinement_reuses_exact_v6_seed_scheme():
    assert _v6_fit_seed(20260812, 0, 1) == 20360813
    assert _v6_fit_seed(20260812, 1, 4) == 20460816


def test_coarse_anchor_produces_one_day_bracket_when_132_fails():
    f = _fine_rows()
    f.loc[f["horizon"].eq(132), "candidate_p_iut"] = 0.20
    c = _coarse_fold_anchor()
    gated = _fixed_sequence_gatekeeping(f, c, upper_anchor=133, alpha=0.05)
    reg = _aggregate_fixed_sequence_regimes(gated, min_folds=3, min_fold_fraction=0.75)
    assert reg[reg.horizon.eq(133)].iloc[0].certified_regime == "R"
    assert reg[reg.horizon.eq(132)].iloc[0].certified_regime == "UNRESOLVED"
    assert reg[reg.horizon.eq(131)].iloc[0].certified_regime == "UNRESOLVED"
    onset = _refined_onset(
        reg,
        _coarse_regimes(),
        return_mode="raw",
        lower_exclusive=119,
        upper_inclusive=133,
        anchor_spacing_days=7,
        min_consecutive=3,
        min_tail_fraction=0.80,
    )
    assert onset["identified"] is True
    assert onset["hc_grid"] == 133
    assert onset["hc_lower_exclusive"] == 132


def test_fixed_sequence_stops_and_does_not_reenter_below_failure():
    f = _fine_rows()
    f.loc[(f["fold"].isin([1, 2, 3])) & f["horizon"].eq(130), "candidate_p_iut"] = 0.20
    c = _coarse_fold_anchor()
    gated = _fixed_sequence_gatekeeping(f, c, upper_anchor=133, alpha=0.05)
    reg = _aggregate_fixed_sequence_regimes(gated, min_folds=3, min_fold_fraction=0.75)
    assert reg[reg.horizon.eq(132)].iloc[0].certified_regime == "R"
    assert reg[reg.horizon.eq(131)].iloc[0].certified_regime == "R"
    assert reg[reg.horizon.eq(130)].iloc[0].certified_regime == "UNRESOLVED"
    assert reg[reg.horizon.eq(129)].iloc[0].certified_regime == "UNRESOLVED"
    onset = _refined_onset(
        reg,
        _coarse_regimes(),
        return_mode="raw",
        lower_exclusive=119,
        upper_inclusive=133,
        anchor_spacing_days=7,
        min_consecutive=3,
        min_tail_fraction=0.80,
    )
    assert onset["hc_grid"] == 131
    assert onset["hc_lower_exclusive"] == 130


def test_anchor_still_requires_original_three_of_four_fold_support():
    f = _fine_rows()
    c = _coarse_fold_anchor(pass_folds=(1, 2))
    gated = _fixed_sequence_gatekeeping(f, c, upper_anchor=133, alpha=0.05)
    reg = _aggregate_fixed_sequence_regimes(gated, min_folds=3, min_fold_fraction=0.75)
    assert reg[reg.horizon.eq(133)].iloc[0].certified_regime == "UNRESOLVED"

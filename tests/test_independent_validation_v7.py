import json
from pathlib import Path

import pandas as pd

from mrspd.critical_horizon_refine import _consensus_refinement, _refined_onset
from mrspd.independent_validation import (
    _assessment,
    _canonical_json_sha256,
    audit_independence,
    freeze_validation_protocol,
)


def _regimes(mode="raw"):
    fine = pd.DataFrame({
        "return_mode": [mode] * 14,
        "horizon": list(range(120, 134)),
        "certified_regime": ["UNRESOLVED"] * 5 + ["R"] * 9,
    })
    coarse = pd.DataFrame({
        "return_mode": [mode] * 4,
        "horizon": [133, 140, 147, 168],
        "certified_regime": ["R", "R", "R", "R"],
    })
    return fine, coarse


def test_refinement_preserves_persistence_time_scale():
    fine, coarse = _regimes()
    got = _refined_onset(
        fine,
        coarse,
        return_mode="raw",
        lower_exclusive=119,
        upper_inclusive=133,
        anchor_spacing_days=7,
        min_consecutive=3,
        min_tail_fraction=0.80,
    )
    # First R is h=125, with persistence anchors 125,132,140.
    assert got["identified"] is True
    assert got["hc_grid"] == 125
    assert got["persistence_anchors"] == [125, 132, 140]
    assert got["hc_lower_exclusive"] == 124


def test_refinement_rejects_nonpersistent_fine_onset():
    fine, coarse = _regimes()
    fine.loc[fine["horizon"] == 132, "certified_regime"] = "UNRESOLVED"
    fine.loc[fine["horizon"] >= 126, "certified_regime"] = "UNRESOLVED"
    got = _refined_onset(
        fine,
        coarse,
        return_mode="raw",
        lower_exclusive=119,
        upper_inclusive=133,
        anchor_spacing_days=7,
        min_consecutive=3,
        min_tail_fraction=0.80,
    )
    assert got["identified"] is False


def test_consensus_is_latest_mode_entry():
    df = pd.DataFrame([
        {"return_mode": "market_residual", "identified": True, "hc_lower_exclusive": 123, "hc_upper_inclusive": 124, "hc_grid": 124},
        {"return_mode": "raw", "identified": True, "hc_lower_exclusive": 128, "hc_upper_inclusive": 129, "hc_grid": 129},
    ])
    c = _consensus_refinement(df)
    assert c["identified"] is True
    assert c["hc_grid"] == 129
    assert c["hc_lower_exclusive"] == 128


def _panel(tickers, start, end):
    dates = pd.date_range(start, end, freq="B")
    rows = []
    for t in tickers:
        for d in dates:
            rows.append({"ticker": t, "date": d, "horizon": 126, "return_mode": "raw"})
    return pd.DataFrame(rows)


def test_universe_independence_requires_zero_overlap():
    d = _panel(["AAA", "BBB"], "2020-01-01", "2020-01-10")
    v = _panel(["CCC", "DDD"], "2020-01-01", "2020-01-10")
    a = audit_independence(d, v, mode="universe")
    assert a["pass"] is True
    v2 = _panel(["BBB", "CCC"], "2020-01-01", "2020-01-10")
    a2 = audit_independence(d, v2, mode="universe")
    assert a2["pass"] is False
    assert a2["ticker_overlap_count"] == 1


def test_time_independence_requires_strictly_later_dates():
    d = _panel(["AAA"], "2020-01-01", "2020-01-10")
    v = _panel(["AAA"], "2020-01-13", "2020-01-20")
    assert audit_independence(d, v, mode="time")["pass"] is True
    v2 = _panel(["AAA"], "2020-01-10", "2020-01-20")
    assert audit_independence(d, v2, mode="time")["pass"] is False


def test_protocol_is_hash_locked(tmp_path: Path):
    discovery = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "horizon": [126, 133],
        "return_mode": ["raw", "market_residual"],
    })
    coarse = {
        "scan_horizons": [63,72,84,96,105,112,119,126,133,140,147,168],
        "cert_alpha": 0.05,
        "cert_min_effect_sigma": 0.02,
        "cert_min_folds": 3,
        "cert_min_fold_fraction": 0.75,
        "min_consecutive_r_horizons": 3,
        "min_r_tail_fraction": 0.8,
        "consensus_hc": {"identified": True, "hc_grid": 133},
    }
    refine = {
        "refinement_horizons": list(range(120,134)),
        "fit_horizons": [63,72,84,96,105,112,119] + list(range(120,134)) + [140,147,168],
        "refinement_interval": {"lower_exclusive": 119, "upper_inclusive": 133},
        "certification_rules_unchanged": {"alpha":0.05,"min_effect_sigma":0.02,"min_folds":3,"min_fold_fraction":0.75,"single_phase_bins":5,"min_consecutive":3,"min_tail_fraction":0.8},
        "consensus_hc_refined": {"identified": True, "hc_grid": 129, "hc_lower_exclusive": 128, "hc_upper_inclusive": 129},
    }
    p = freeze_validation_protocol(
        discovery_panel=discovery,
        discovery_coarse_summary=coarse,
        discovery_refinement_summary=refine,
        out_path=tmp_path / "protocol.json",
    )
    q = dict(p)
    h = q.pop("protocol_sha256")
    assert h == _canonical_json_sha256(q)
    assert p["discovery"]["target_hc_grid"] == 129
    assert p["primary_endpoints"]["localization_tolerance_days"] == 7


def test_confirmation_assessment_requires_all_three_parts():
    protocol = {
        "discovery": {"target_hc_grid": 129},
        "primary_endpoints": {"localization_tolerance_days": 7},
    }
    coarse = {"consensus_hc": {"identified": True}}
    refine = {"consensus_hc_refined": {"identified": True, "hc_grid": 133}}
    got = _assessment(protocol, coarse, refine, {"pass": True})
    assert got["overall_confirmatory_pass"] is True
    refine_bad = {"consensus_hc_refined": {"identified": True, "hc_grid": 147}}
    got2 = _assessment(protocol, coarse, refine_bad, {"pass": True})
    assert got2["overall_confirmatory_pass"] is False

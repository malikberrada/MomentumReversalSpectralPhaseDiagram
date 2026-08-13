from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mrspd import independent_validation as iv

COARSE = [63,72,84,96,105,112,119,126,133,140,147,168]
FINE = list(range(120,134))
MEASUREMENT = sorted(set(COARSE).union(FINE))


def _discovery_panel() -> pd.DataFrame:
    rows = []
    for mode in ["raw", "market_residual"]:
        for h in MEASUREMENT:
            rows.append({"ticker": "AAA", "date": pd.Timestamp("2026-01-02"), "horizon": h, "return_mode": mode, "psi_primary": 0.0, "phase_product": 0.0})
    return pd.DataFrame(rows)


def _coarse_summary() -> dict:
    return {
        "scan_horizons": COARSE,
        "cert_alpha": 0.05,
        "cert_min_effect_sigma": 0.02,
        "cert_min_folds": 3,
        "cert_min_fold_fraction": 0.75,
        "min_consecutive_r_horizons": 3,
        "min_r_tail_fraction": 0.8,
        "consensus_hc": {"identified": True, "hc_grid": 133},
    }


def _refine_summary() -> dict:
    return {
        "status": "EXPLORATORY_RESOLUTION_REFINEMENT_ONLY_V8_METHOD_CORRECTION",
        "refinement_horizons": FINE,
        "measurement_panel_horizons": MEASUREMENT,
        # v8 compatibility field may still contain dense rows; it must NOT be used for fitting.
        "fit_horizons": MEASUREMENT,
        "surface_fit_horizons": COARSE,
        "refinement_interval": {"lower_exclusive": 119, "upper_inclusive": 133},
        "certification_rules_unchanged": {
            "spline_knots": 7,
            "bootstrap_reps": 1000,
            "max_fit_rows": 250000,
            "alpha": 0.05,
            "min_effect_sigma": 0.02,
            "abs_effect": 0.0,
            "min_obs_per_band": 250,
            "min_dates_per_band": 40,
            "min_folds": 3,
            "min_fold_fraction": 0.75,
            "single_phase_bins": 5,
            "min_consecutive": 3,
            "min_tail_fraction": 0.8,
            "walkforward_splits": 4,
            "min_train_frac": 0.5,
            "stride": 5,
            "seed": 20260812,
        },
        "consensus_hc_refined": {
            "identified": True,
            "hc_grid": 128,
            "hc_lower_exclusive": 127,
            "hc_upper_inclusive": 128,
        },
    }


def test_freeze_v8_separates_surface_fit_from_measurement(tmp_path: Path):
    p = iv.freeze_validation_protocol(
        discovery_panel=_discovery_panel(),
        discovery_coarse_summary=_coarse_summary(),
        discovery_refinement_summary=_refine_summary(),
        out_path=tmp_path / "protocol.json",
    )
    assert p["protocol_version"] == "MRSPD-HC-INDEPENDENT-VALIDATION-v8"
    assert p["surface_fit_horizons"] == COARSE
    assert p["measurement_horizons"] == MEASUREMENT
    assert "fit_horizons" not in p
    assert set(p["refinement_horizons"]).issubset(set(p["measurement_horizons"]))
    assert set(p["surface_fit_horizons"]) == set(p["coarse_horizons"])


def test_freeze_rejects_dense_surface_fit(tmp_path: Path):
    r = _refine_summary()
    r["surface_fit_horizons"] = MEASUREMENT
    with pytest.raises(ValueError, match="surface_fit_horizons must exactly equal"):
        iv.freeze_validation_protocol(
            discovery_panel=_discovery_panel(),
            discovery_coarse_summary=_coarse_summary(),
            discovery_refinement_summary=r,
            out_path=tmp_path / "bad.json",
        )


def test_validate_protocol_rejects_legacy_fit_horizons_only():
    p = {
        "protocol_version": iv.PROTOCOL_VERSION,
        "coarse_horizons": COARSE,
        "refinement_horizons": FINE,
        "fit_horizons": MEASUREMENT,
    }
    with pytest.raises(ValueError, match="surface_fit_horizons"):
        iv._validate_protocol_horizon_design(p)


def test_validate_protocol_rejects_dense_fit_contamination():
    p = {
        "protocol_version": iv.PROTOCOL_VERSION,
        "coarse_horizons": COARSE,
        "refinement_horizons": FINE,
        "surface_fit_horizons": MEASUREMENT,
        "measurement_horizons": MEASUREMENT,
    }
    with pytest.raises(ValueError, match="exactly equal the frozen coarse_horizons"):
        iv._validate_protocol_horizon_design(p)


def test_validation_routes_dense_horizons_to_measurement_only(tmp_path: Path, monkeypatch):
    protocol = iv.freeze_validation_protocol(
        discovery_panel=_discovery_panel(),
        discovery_coarse_summary=_coarse_summary(),
        discovery_refinement_summary=_refine_summary(),
        out_path=tmp_path / "protocol.json",
    )
    # Avoid large computations and isolate routing semantics.
    monkeypatch.setattr(iv, "audit_independence", lambda *a, **k: {"pass": True})
    monkeypatch.setattr(iv, "_verify_discovery_against_protocol", lambda *a, **k: {"pass": True, "checks": {}})

    calls = {}
    def fake_coarse(panel, cfg, out_dir, **kwargs):
        calls["coarse_panel_h"] = sorted(set(panel["horizon"].astype(int)))
        calls["coarse_cfg_h"] = list(cfg.horizons)
        return {"consensus_hc": {"identified": True, "hc_grid": 133}}

    def fake_refine(panel, cfg, coarse_dir, out_dir, **kwargs):
        calls["refine_panel_h"] = sorted(set(panel["horizon"].astype(int)))
        calls["refine_cfg_h"] = list(cfg.horizons)
        calls["refine_fit_h"] = list(kwargs["fit_horizons"])
        calls["refine_measure_h"] = list(kwargs["refinement_horizons"])
        return {"consensus_hc_refined": {"identified": True, "hc_grid": 128}}

    monkeypatch.setattr(iv, "run_critical_horizon_scan", fake_coarse)
    monkeypatch.setattr(iv, "run_hc_refinement", fake_refine)
    monkeypatch.setattr(iv, "_assessment", lambda *a, **k: {"overall_confirmatory_pass": True})

    validation = _discovery_panel().copy()
    validation["ticker"] = "ZZZ"
    iv.run_independent_validation(
        discovery_panel=_discovery_panel(),
        validation_panel=validation,
        protocol=protocol,
        out_dir=tmp_path / "validation",
        independence_mode="universe",
    )

    assert calls["coarse_panel_h"] == COARSE
    assert calls["coarse_cfg_h"] == COARSE
    assert calls["refine_panel_h"] == MEASUREMENT
    assert calls["refine_cfg_h"] == COARSE
    assert calls["refine_fit_h"] == COARSE
    assert calls["refine_measure_h"] == FINE

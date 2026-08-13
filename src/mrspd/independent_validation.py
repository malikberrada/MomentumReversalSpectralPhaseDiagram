from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from .critical_horizon import DEFAULT_CRITICAL_HORIZONS, run_critical_horizon_scan
from .critical_horizon_refine import (
    DEFAULT_REFINEMENT_FIT_HORIZONS,
    DEFAULT_REFINEMENT_HORIZONS,
    run_hc_refinement,
)


PROTOCOL_VERSION = "MRSPD-HC-INDEPENDENT-VALIDATION-v8"

PANEL_AUDIT_COLUMNS = ("date", "ticker", "return_mode", "horizon")
PANEL_ANALYSIS_COLUMNS = ("date", "return_mode", "horizon", "psi_primary", "phase_product")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(doc: dict) -> str:
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return _sha256_bytes(payload)


def _panel_metadata(panel: pd.DataFrame) -> dict:
    """Compute metadata without copying/consolidating a multi-million-row panel."""
    missing = [c for c in PANEL_AUDIT_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel is missing audit columns {missing}")
    dates = panel["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
        dates = pd.to_datetime(dates)
    # Unique first, stringify second: never allocate a 30M-row object array just
    # to build a set of a few hundred ticker/mode values.
    tickers = sorted(str(x) for x in panel["ticker"].dropna().unique())
    horizons = sorted(int(x) for x in panel["horizon"].dropna().unique())
    modes = sorted(str(x) for x in panel["return_mode"].dropna().unique())
    meta = {
        "rows": int(len(panel)),
        "ticker_count": int(len(tickers)),
        "tickers": tickers,
        "ticker_set_sha256": _sha256_bytes("\n".join(tickers).encode("utf-8")),
        "date_min": str(dates.min().date()) if len(panel) else None,
        "date_max": str(dates.max().date()) if len(panel) else None,
        "horizons": horizons,
        "return_modes": modes,
    }
    meta["metadata_sha256"] = _canonical_json_sha256({k: v for k, v in meta.items() if k != "tickers"})
    return meta


def stream_panel_metadata(path: Path, *, chunksize: int = 250_000) -> dict:
    """Compute panel identity/audit metadata in bounded memory from CSV/CSV.GZ.

    This intentionally avoids materializing the full discovery/validation panel.
    Pandas' C tokenizer can otherwise exhaust RAM on 20M-30M row gzip files even
    when `usecols` is small. Chunking bounds tokenizer and DataFrame residency.
    The returned schema/hash is identical to `_panel_metadata`.
    """
    path = Path(path)
    tickers: set[str] = set()
    horizons: set[int] = set()
    modes: set[str] = set()
    rows = 0
    date_min = None
    date_max = None
    dtype = {
        "ticker": "string",
        "return_mode": "string",
        "horizon": "int16",
    }
    for chunk in pd.read_csv(
        path,
        usecols=list(PANEL_AUDIT_COLUMNS),
        dtype=dtype,
        chunksize=int(chunksize),
        low_memory=True,
    ):
        rows += int(len(chunk))
        tickers.update(str(x) for x in chunk["ticker"].dropna().unique())
        horizons.update(int(x) for x in chunk["horizon"].dropna().unique())
        modes.update(str(x) for x in chunk["return_mode"].dropna().unique())
        dates = pd.to_datetime(chunk["date"], errors="raise")
        if len(dates):
            cmin = dates.min()
            cmax = dates.max()
            if pd.notna(cmin) and (date_min is None or cmin < date_min):
                date_min = cmin
            if pd.notna(cmax) and (date_max is None or cmax > date_max):
                date_max = cmax

    ticker_list = sorted(tickers)
    meta = {
        "rows": int(rows),
        "ticker_count": int(len(ticker_list)),
        "tickers": ticker_list,
        "ticker_set_sha256": _sha256_bytes("\n".join(ticker_list).encode("utf-8")),
        "date_min": str(date_min.date()) if date_min is not None else None,
        "date_max": str(date_max.date()) if date_max is not None else None,
        "horizons": sorted(horizons),
        "return_modes": sorted(modes),
    }
    meta["metadata_sha256"] = _canonical_json_sha256(
        {k: v for k, v in meta.items() if k != "tickers"}
    )
    return meta


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_validation_protocol(
    *,
    discovery_panel: pd.DataFrame,
    discovery_coarse_summary: dict,
    discovery_refinement_summary: dict,
    out_path: Path,
) -> dict:
    """Freeze all confirmatory settings before independent validation is read."""
    panel_meta = _panel_metadata(discovery_panel)
    refined = discovery_refinement_summary.get("consensus_hc_refined", {}) or {}
    coarse = discovery_coarse_summary.get("consensus_hc", {}) or {}
    target = refined if refined.get("identified") else coarse
    if not target.get("identified"):
        raise ValueError("Discovery h_c was not identified; cannot freeze localization confirmation")

    refinement_rules = discovery_refinement_summary.get("certification_rules_unchanged", {})
    coarse_horizons = tuple(int(x) for x in discovery_coarse_summary.get("scan_horizons", DEFAULT_CRITICAL_HORIZONS))
    fine_horizons = tuple(int(x) for x in discovery_refinement_summary.get("refinement_horizons", DEFAULT_REFINEMENT_HORIZONS))

    # v8/v8.1 methodological invariant: dense refinement horizons are
    # measurement-only. The continuous surface/root-free gate must be fit on
    # exactly the frozen v6 coarse design. Do NOT fall back to the legacy
    # `fit_horizons` field because v7/v8 compatibility summaries may contain
    # dense horizons there.
    surface_fit_horizons = tuple(
        int(x) for x in discovery_refinement_summary.get("surface_fit_horizons", coarse_horizons)
    )
    measurement_horizons = tuple(
        int(x)
        for x in discovery_refinement_summary.get(
            "measurement_panel_horizons",
            sorted(set(coarse_horizons).union(fine_horizons)),
        )
    )
    if tuple(sorted(set(surface_fit_horizons))) != tuple(sorted(set(coarse_horizons))):
        raise ValueError(
            "surface_fit_horizons must exactly equal the frozen coarse_horizons; "
            "dense refinement horizons are measurement-only"
        )
    required_measurement = set(coarse_horizons).union(fine_horizons)
    if not required_measurement.issubset(set(measurement_horizons)):
        missing = sorted(required_measurement - set(measurement_horizons))
        raise ValueError(f"measurement_horizons is missing required horizons {missing}")

    # One original local coarse-grid step. This is frozen before validation and
    # is not tuned against the independent sample.
    coarse_local = sorted(h for h in coarse_horizons if 112 <= h <= 140)
    diffs = [b - a for a, b in zip(coarse_local[:-1], coarse_local[1:])]
    localization_tolerance_days = int(round(float(np.median(diffs)))) if diffs else 7

    rules = {
        "spline_knots": int(refinement_rules.get("spline_knots", 7)),
        "bootstrap_reps": int(refinement_rules.get("bootstrap_reps", 1000)),
        "max_fit_rows": int(refinement_rules.get("max_fit_rows", 250000)),
        "cert_alpha": float(refinement_rules.get("alpha", discovery_coarse_summary.get("cert_alpha", 0.05))),
        "cert_min_effect_sigma": float(refinement_rules.get("min_effect_sigma", discovery_coarse_summary.get("cert_min_effect_sigma", 0.02))),
        "cert_abs_effect": float(refinement_rules.get("abs_effect", 0.0)),
        "cert_min_obs_per_band": int(refinement_rules.get("min_obs_per_band", 250)),
        "cert_min_dates_per_band": int(refinement_rules.get("min_dates_per_band", 40)),
        "cert_min_folds": int(refinement_rules.get("min_folds", discovery_coarse_summary.get("cert_min_folds", 3))),
        "cert_min_fold_fraction": float(refinement_rules.get("min_fold_fraction", discovery_coarse_summary.get("cert_min_fold_fraction", 0.75))),
        "cert_single_phase_bins": int(refinement_rules.get("single_phase_bins", 5)),
        "hc_min_consecutive": int(refinement_rules.get("min_consecutive", discovery_coarse_summary.get("min_consecutive_r_horizons", 3))),
        "hc_min_tail_fraction": float(refinement_rules.get("min_tail_fraction", discovery_coarse_summary.get("min_r_tail_fraction", 0.80))),
        "walkforward_splits": int(refinement_rules.get("walkforward_splits", 4)),
        "min_train_frac": float(refinement_rules.get("min_train_frac", 0.50)),
        "stride": int(refinement_rules.get("stride", 5)),
        "seed": int(refinement_rules.get("seed", 20260812)),
    }

    target_grid = int(target["hc_grid"])
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_BEFORE_INDEPENDENT_VALIDATION",
        "discovery": {
            "panel_metadata": panel_meta,
            "coarse_consensus_hc": coarse,
            "refined_consensus_hc": refined,
            "target_hc_grid": target_grid,
        },
        "independence_allowed": ["universe", "time", "both"],
        "coarse_horizons": list(coarse_horizons),
        "refinement_horizons": list(fine_horizons),
        "surface_fit_horizons": list(surface_fit_horizons),
        "measurement_horizons": list(measurement_horizons),
        "refinement_interval": discovery_refinement_summary.get("refinement_interval", {
            "lower_exclusive": min(fine_horizons) - 1,
            "upper_inclusive": max(fine_horizons),
        }),
        "rules": rules,
        "primary_endpoints": {
            "phenomenon": "persistent OOS-certified R-only critical tail is independently identified in both raw and market_residual",
            "localization": "independent refined consensus h_c is within one frozen original coarse-grid step of discovery target",
            "localization_tolerance_days": localization_tolerance_days,
            "overall_pass_requires": ["independence_audit", "phenomenon", "localization"],
        },
        "guardrails": [
            "No validation thresholds may be supplied by the confirmation CLI; all are loaded from this frozen protocol.",
            "Universe independence requires zero ticker overlap between discovery and validation panels.",
            "Time independence requires validation panel dates to start strictly after the discovery panel end date.",
            "The refined h_c bracket is a discrete grid bracket, not a confidence interval.",
            "The confirmatory spline/root-free gate is fit on surface_fit_horizons only; dense refinement horizons are measurement-only.",
        ],
    }
    protocol["protocol_sha256"] = _canonical_json_sha256(protocol)
    Path(out_path).write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")
    return protocol


def audit_independence_metadata(
    discovery_metadata: dict,
    validation_metadata: dict,
    *,
    mode: str,
) -> dict:
    """Audit independence from compact metadata only (no full-panel copies)."""
    mode = str(mode).lower()
    if mode not in {"universe", "time", "both"}:
        raise ValueError("independence mode must be universe, time, or both")
    dt = {str(x) for x in discovery_metadata.get("tickers", [])}
    vt = {str(x) for x in validation_metadata.get("tickers", [])}
    overlap = sorted(dt.intersection(vt))
    discovery_end = pd.to_datetime(discovery_metadata.get("date_max"), errors="coerce")
    validation_start = pd.to_datetime(validation_metadata.get("date_min"), errors="coerce")
    universe_pass = len(overlap) == 0
    time_pass = bool(
        pd.notna(discovery_end)
        and pd.notna(validation_start)
        and validation_start > discovery_end
    )
    required = (
        universe_pass
        if mode == "universe"
        else time_pass
        if mode == "time"
        else (universe_pass and time_pass)
    )
    return {
        "mode": mode,
        "pass": bool(required),
        "universe_pass": bool(universe_pass),
        "ticker_overlap_count": int(len(overlap)),
        "ticker_overlap_preview": overlap[:20],
        "time_pass": bool(time_pass),
        "discovery_end": str(discovery_end.date()) if pd.notna(discovery_end) else None,
        "validation_start": str(validation_start.date()) if pd.notna(validation_start) else None,
    }


def audit_independence(
    discovery_panel: pd.DataFrame,
    validation_panel: pd.DataFrame,
    *,
    mode: str,
) -> dict:
    """Backward-compatible DataFrame entrypoint; internally audit metadata only."""
    return audit_independence_metadata(
        _panel_metadata(discovery_panel),
        _panel_metadata(validation_panel),
        mode=mode,
    )


def _verify_discovery_metadata(got: dict, protocol: dict) -> dict:
    exp = protocol["discovery"]["panel_metadata"]
    checks = {
        "ticker_set_sha256": got["ticker_set_sha256"] == exp["ticker_set_sha256"],
        "date_max": got["date_max"] == exp["date_max"],
        "horizons_contain_protocol_coarse": set(protocol["coarse_horizons"]).issubset(
            set(got["horizons"])
        ),
    }
    return {"pass": all(checks.values()), "checks": checks, "current": got}

def _verify_discovery_against_protocol(discovery_panel: pd.DataFrame, protocol: dict) -> dict:
    return _verify_discovery_metadata(_panel_metadata(discovery_panel), protocol)


def _assessment(protocol: dict, coarse_summary: dict, refine_summary: dict, independence: dict) -> dict:
    coarse_cons = coarse_summary.get("consensus_hc", {}) or {}
    refined_cons = refine_summary.get("consensus_hc_refined", {}) or {}
    phenomenon = bool(coarse_cons.get("identified"))
    target = int(protocol["discovery"]["target_hc_grid"])
    tol = int(protocol["primary_endpoints"]["localization_tolerance_days"])
    val_h = refined_cons.get("hc_grid")
    localization = bool(refined_cons.get("identified") and val_h is not None and abs(int(val_h) - target) <= tol)
    overall = bool(independence.get("pass") and phenomenon and localization)
    return {
        "independence_pass": bool(independence.get("pass")),
        "phenomenon_confirmed": phenomenon,
        "localization_confirmed": localization,
        "discovery_target_hc_grid": target,
        "validation_refined_hc_grid": int(val_h) if val_h is not None else None,
        "localization_tolerance_days": tol,
        "overall_confirmatory_pass": overall,
        "interpretation": "CONFIRMED" if overall else "NOT_CONFIRMED",
    }


def _validate_protocol_horizon_design(protocol: dict) -> dict:
    """Fail closed if a confirmatory protocol can contaminate the coarse fit.

    v8 confirmation requires explicit separation between the frozen coarse
    surface-fit design and the dense measurement grid. Legacy `fit_horizons`
    alone is deliberately rejected rather than guessed.
    """
    if "surface_fit_horizons" not in protocol:
        raise ValueError(
            "v8 validation protocol is missing surface_fit_horizons; legacy fit_horizons-only protocols are not confirmatory-safe"
        )
    if "measurement_horizons" not in protocol:
        raise ValueError("v8 validation protocol is missing measurement_horizons")
    coarse = tuple(sorted(set(int(x) for x in protocol.get("coarse_horizons", []))))
    surface = tuple(sorted(set(int(x) for x in protocol.get("surface_fit_horizons", []))))
    fine = tuple(sorted(set(int(x) for x in protocol.get("refinement_horizons", []))))
    measurement = tuple(sorted(set(int(x) for x in protocol.get("measurement_horizons", []))))
    if not coarse:
        raise ValueError("v8 validation protocol has no coarse_horizons")
    if surface != coarse:
        raise ValueError(
            "surface_fit_horizons must exactly equal the frozen coarse_horizons; dense refinement horizons are measurement-only"
        )
    required = set(coarse).union(fine)
    missing = sorted(required - set(measurement))
    if missing:
        raise ValueError(f"measurement_horizons is missing required horizons {missing}")
    dense_only = sorted(set(fine) - set(coarse))
    contaminated = sorted(set(dense_only).intersection(surface))
    if contaminated:
        raise ValueError(
            f"dense refinement horizons leaked into surface_fit_horizons: {contaminated}"
        )
    return {
        "surface_fit_horizons": list(surface),
        "measurement_horizons": list(measurement),
        "dense_measurement_only_horizons": dense_only,
        "pass": True,
    }


def run_independent_validation(
    *,
    discovery_panel: pd.DataFrame | None,
    validation_panel: pd.DataFrame,
    protocol: dict,
    out_dir: Path,
    independence_mode: str,
    discovery_metadata: dict | None = None,
    validation_metadata: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported or missing validation protocol version")
    expected_hash = protocol.get("protocol_sha256")
    tmp = dict(protocol)
    tmp.pop("protocol_sha256", None)
    if expected_hash != _canonical_json_sha256(tmp):
        raise ValueError("Validation protocol SHA256 mismatch; protocol may have been edited")

    horizon_design = _validate_protocol_horizon_design(protocol)
    (out_dir / "confirmatory_horizon_design_audit.json").write_text(
        json.dumps(horizon_design, indent=2), encoding="utf-8"
    )

    if discovery_metadata is None:
        if discovery_panel is None:
            raise ValueError("discovery_panel or discovery_metadata is required")
        discovery_metadata = _panel_metadata(discovery_panel)
    if validation_metadata is None:
        # Backward-compatible path for small/in-memory callers. The streaming CLI
        # supplies validation_metadata because its scientific frame intentionally
        # omits ticker after the audit pass.
        required_audit = set(PANEL_AUDIT_COLUMNS)
        if required_audit.issubset(validation_panel.columns):
            validation_metadata = _panel_metadata(validation_panel)
        else:
            raise ValueError(
                "validation_metadata is required when validation_panel omits audit columns"
            )

    discovery_check = _verify_discovery_metadata(discovery_metadata, protocol)
    if not discovery_check["pass"]:
        raise ValueError(
            f"Discovery panel does not match frozen protocol: {discovery_check['checks']}"
        )

    independence = audit_independence_metadata(
        discovery_metadata, validation_metadata, mode=independence_mode
    )
    (out_dir / "independence_audit.json").write_text(
        json.dumps(independence, indent=2), encoding="utf-8"
    )
    if not independence["pass"]:
        raise ValueError(f"Independent validation rejected by audit: {independence}")

    # Full discovery data are never needed in the streaming CLI.
    discovery_panel = None
    gc.collect()

    missing_analysis = [c for c in PANEL_ANALYSIS_COLUMNS if c not in validation_panel.columns]
    if missing_analysis:
        raise ValueError(f"Validation panel is missing analysis columns {missing_analysis}")
    # Keep only the five columns used by the critical-horizon algorithms.
    # The CLI already loads compact dtypes; this shallow column selection avoids
    # carrying ticker/unused OHLCV/spectral columns through repeated folds.
    validation_panel = validation_panel.loc[:, list(PANEL_ANALYSIS_COLUMNS)]
    gc.collect()

    rules = protocol["rules"]
    coarse_h = tuple(int(x) for x in protocol["coarse_horizons"])
    fine_h = tuple(int(x) for x in protocol["refinement_horizons"])
    surface_fit_h = tuple(int(x) for x in protocol["surface_fit_horizons"])
    measurement_h = tuple(int(x) for x in protocol["measurement_horizons"])
    needed = set(measurement_h)
    available = set(int(x) for x in validation_panel["horizon"].dropna().unique())
    missing = sorted(needed - available)
    if missing:
        raise ValueError(
            "Validation panel is missing frozen protocol measurement horizons " + str(missing)
        )

    # Keep cfg's horizon design coarse-only. Dense rows may be present in the
    # measurement panel, but they are never allowed to change the surface fit.
    cfg = SimpleNamespace(
        horizons=surface_fit_h,
        spectral_spans=surface_fit_h,
        min_train_frac=float(rules["min_train_frac"]),
        n_walkforward_splits=int(rules["walkforward_splits"]),
        anchor_stride=int(rules["stride"]),
        random_seed=int(rules["seed"]),
    )

    coarse_out = out_dir / "coarse"
    coarse_summary = run_critical_horizon_scan(
        validation_panel.loc[validation_panel["horizon"].isin(coarse_h), :],
        SimpleNamespace(
            horizons=coarse_h,
            spectral_spans=coarse_h,
            min_train_frac=cfg.min_train_frac,
            n_walkforward_splits=cfg.n_walkforward_splits,
            anchor_stride=cfg.anchor_stride,
            random_seed=cfg.random_seed,
        ),
        coarse_out,
        horizons=coarse_h,
        spline_knots=int(rules["spline_knots"]),
        bootstrap_reps=int(rules["bootstrap_reps"]),
        max_fit_rows=int(rules["max_fit_rows"]),
        cert_alpha=float(rules["cert_alpha"]),
        cert_min_effect_sigma=float(rules["cert_min_effect_sigma"]),
        cert_abs_effect=float(rules["cert_abs_effect"]),
        cert_min_obs_per_band=int(rules["cert_min_obs_per_band"]),
        cert_min_dates_per_band=int(rules["cert_min_dates_per_band"]),
        cert_min_folds=int(rules["cert_min_folds"]),
        cert_min_fold_fraction=float(rules["cert_min_fold_fraction"]),
        cert_single_phase_bins=int(rules["cert_single_phase_bins"]),
        hc_min_consecutive=int(rules["hc_min_consecutive"]),
        hc_min_tail_fraction=float(rules["hc_min_tail_fraction"]),
    )

    ref_interval = protocol["refinement_interval"]
    refine_out = out_dir / "refinement"
    refine_summary = run_hc_refinement(
        validation_panel.loc[validation_panel["horizon"].isin(measurement_h), :],
        cfg,
        coarse_out,
        refine_out,
        refinement_horizons=fine_h,
        fit_horizons=surface_fit_h,
        lower_exclusive=int(ref_interval["lower_exclusive"]),
        upper_inclusive=int(ref_interval["upper_inclusive"]),
        spline_knots=int(rules["spline_knots"]),
        bootstrap_reps=int(rules["bootstrap_reps"]),
        max_fit_rows=int(rules["max_fit_rows"]),
        cert_alpha=float(rules["cert_alpha"]),
        cert_min_effect_sigma=float(rules["cert_min_effect_sigma"]),
        cert_abs_effect=float(rules["cert_abs_effect"]),
        cert_min_obs_per_band=int(rules["cert_min_obs_per_band"]),
        cert_min_dates_per_band=int(rules["cert_min_dates_per_band"]),
        cert_min_folds=int(rules["cert_min_folds"]),
        cert_min_fold_fraction=float(rules["cert_min_fold_fraction"]),
        cert_single_phase_bins=int(rules["cert_single_phase_bins"]),
        hc_min_consecutive=int(rules["hc_min_consecutive"]),
        hc_min_tail_fraction=float(rules["hc_min_tail_fraction"]),
    )

    assessment = _assessment(protocol, coarse_summary, refine_summary, independence)
    result = {
        "status": "INDEPENDENT_CONFIRMATORY_VALIDATION",
        "protocol_sha256": protocol["protocol_sha256"],
        "independence": independence,
        "horizon_design_audit": horizon_design,
        "coarse_summary": coarse_summary,
        "refinement_summary": refine_summary,
        "assessment": assessment,
        "guardrail": "A failed independent confirmation must not be converted to PASS by retuning thresholds on the same validation sample.",
    }
    (out_dir / "independent_validation_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame([assessment]).to_csv(out_dir / "independent_validation_assessment.csv", index=False)
    return result

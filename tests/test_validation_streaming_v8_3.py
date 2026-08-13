from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrspd import independent_validation as iv
from mrspd import independent_validation_cli as cli


def _wide_panel() -> pd.DataFrame:
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        for mode in ["market_residual", "raw"]:
            for horizon in [63, 126, 133]:
                rows.append(
                    {
                        "date": "2026-01-02",
                        "ticker": ticker,
                        "asset_class": "us_stock",
                        "return_mode": mode,
                        "horizon": horizon,
                        "psi_primary": 0.125,
                        "phase_product": -0.25,
                        "psi_63": 1.0,
                        "psi_126": 2.0,
                        "vol20": 3.0,
                        "range20": 4.0,
                        "log_dollar_volume20": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_stream_metadata_matches_in_memory_metadata(tmp_path: Path) -> None:
    df = _wide_panel()
    p = tmp_path / "panel.csv.gz"
    df.to_csv(p, index=False, compression="gzip")
    streamed = iv.stream_panel_metadata(p, chunksize=5)
    expected = iv._panel_metadata(df.loc[:, list(iv.PANEL_AUDIT_COLUMNS)])
    assert streamed == expected


def test_streaming_analysis_loader_preserves_scientific_dtypes(tmp_path: Path) -> None:
    df = _wide_panel()
    p = tmp_path / "panel.csv.gz"
    df.to_csv(p, index=False, compression="gzip")
    got = cli._read_analysis_panel_streaming(p, chunksize=4)
    assert list(got.columns) == list(iv.PANEL_ANALYSIS_COLUMNS)
    assert str(got["date"].dtype) == "datetime64[ns]"
    assert str(got["return_mode"].dtype) == "category"
    assert str(got["horizon"].dtype) == "int16"
    assert str(got["psi_primary"].dtype) == "float64"
    assert str(got["phase_product"].dtype) == "float64"
    assert len(got) == len(df)


def test_metadata_only_universe_audit() -> None:
    d = {
        "tickers": ["AAA", "BBB"],
        "date_max": "2026-05-11",
    }
    v = {
        "tickers": ["CCC", "DDD"],
        "date_min": "2007-11-27",
    }
    out = iv.audit_independence_metadata(d, v, mode="universe")
    assert out["pass"] is True
    assert out["universe_pass"] is True
    assert out["time_pass"] is False
    assert out["ticker_overlap_count"] == 0


def test_v82_compat_loader_still_exists(tmp_path: Path) -> None:
    df = _wide_panel()
    p = tmp_path / "panel.csv.gz"
    df.to_csv(p, index=False, compression="gzip")
    got = cli._read_compact_panel(p, validation=True)
    assert "ticker" in got.columns
    assert str(got["ticker"].dtype) == "category"
    assert str(got["psi_primary"].dtype) == "float64"

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrspd import independent_validation_cli as cli


def test_compact_compat_loader_forces_nanosecond_dates(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "ticker": ["AAA", "BBB"],
            "return_mode": ["raw", "market_residual"],
            "horizon": [126, 133],
            "psi_primary": [0.1, 0.2],
            "phase_product": [-0.2, -0.3],
        }
    )
    path = tmp_path / "panel.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")

    got = cli._read_compact_panel(path, validation=True)
    assert str(got["date"].dtype) == "datetime64[ns]"


def test_streaming_loader_forces_nanosecond_dates_after_concat(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "ticker": ["AAA", "BBB", "CCC"],
            "return_mode": ["raw", "market_residual", "raw"],
            "horizon": [126, 133, 140],
            "psi_primary": [0.1, 0.2, 0.3],
            "phase_product": [-0.2, -0.3, -0.4],
        }
    )
    path = tmp_path / "panel.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")

    got = cli._read_analysis_panel_streaming(path, chunksize=1)
    assert str(got["date"].dtype) == "datetime64[ns]"

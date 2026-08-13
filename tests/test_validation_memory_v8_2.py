from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mrspd import independent_validation as iv
from mrspd import independent_validation_cli as cli


def _small_panel(tickers: list[str], *, validation: bool) -> pd.DataFrame:
    rows = []
    for t in tickers:
        for h in [63, 126, 133]:
            row = {
                "date": pd.Timestamp("2026-01-02"),
                "ticker": t,
                "return_mode": "raw",
                "horizon": h,
            }
            if validation:
                row.update({"psi_primary": 0.1, "phase_product": -0.2})
            rows.append(row)
    z = pd.DataFrame(rows)
    z["ticker"] = z["ticker"].astype("category")
    z["return_mode"] = z["return_mode"].astype("category")
    z["horizon"] = z["horizon"].astype("int16")
    return z


def test_audit_independence_never_copies_full_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    d = _small_panel(["AAA", "BBB"], validation=False)
    v = _small_panel(["CCC", "DDD"], validation=True)

    def forbidden_copy(self, *args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("DataFrame.copy() is forbidden inside audit_independence")

    monkeypatch.setattr(pd.DataFrame, "copy", forbidden_copy)
    out = iv.audit_independence(d, v, mode="universe")
    assert out["pass"] is True
    assert out["ticker_overlap_count"] == 0


def test_panel_metadata_unique_first_and_category_safe() -> None:
    d = _small_panel(["AAA", "BBB"], validation=False)
    meta = iv._panel_metadata(d)
    assert meta["ticker_count"] == 2
    assert meta["tickers"] == ["AAA", "BBB"]
    assert meta["horizons"] == [63, 126, 133]


def test_compact_loader_columns_and_dtypes(tmp_path: Path) -> None:
    full = _small_panel(["CCC", "DDD"], validation=True)
    # Add columns representative of the wide 38-column production panel.
    full["psi_63"] = 1.0
    full["vol20"] = 2.0
    full["range20"] = 3.0
    full.to_csv(tmp_path / "panel.csv.gz", index=False, compression="gzip")

    audit = cli._read_compact_panel(tmp_path / "panel.csv.gz", validation=False)
    val = cli._read_compact_panel(tmp_path / "panel.csv.gz", validation=True)

    assert list(audit.columns) == list(iv.PANEL_AUDIT_COLUMNS)
    assert set(val.columns) == set(iv.PANEL_AUDIT_COLUMNS).union(iv.PANEL_ANALYSIS_COLUMNS)
    assert "psi_63" not in val.columns
    assert "vol20" not in val.columns
    assert str(val["ticker"].dtype) == "category"
    assert str(val["return_mode"].dtype) == "category"
    assert str(val["horizon"].dtype) == "int16"
    assert str(val["psi_primary"].dtype) == "float64"
    assert str(val["phase_product"].dtype) == "float64"

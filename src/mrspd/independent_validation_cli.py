from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from .independent_validation import (
    PANEL_ANALYSIS_COLUMNS,
    run_independent_validation,
    stream_panel_metadata,
)


DEFAULT_CHUNKSIZE = 250_000
EXECUTION_PATCH_VERSION = "8.3.1"
_RETURN_MODE_CATEGORIES = ["market_residual", "raw"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MRSPD h_c validation on a genuinely independent panel using a frozen protocol"
    )
    p.add_argument("--protocol", required=True)
    p.add_argument("--discovery-panel", required=True)
    p.add_argument("--validation-panel", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--independence", choices=["universe", "time", "both"], required=True)
    return p.parse_args()


def _read_compact_panel(path: Path, *, validation: bool) -> pd.DataFrame:
    """Backward-compatible bounded-memory loader used by v8.2 tests/tools.

    The production confirmatory CLI no longer uses this for discovery. It is
    retained so existing tests and small callers do not break after v8.3.
    """
    from .independent_validation import PANEL_AUDIT_COLUMNS

    usecols = list(PANEL_AUDIT_COLUMNS)
    if validation:
        usecols += [c for c in PANEL_ANALYSIS_COLUMNS if c not in usecols]
    dtype = {
        "ticker": "string",
        "return_mode": "string",
        "horizon": "int16",
    }
    if validation:
        dtype.update({"psi_primary": "float64", "phase_product": "float64"})
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        Path(path),
        usecols=usecols,
        dtype=dtype,
        chunksize=DEFAULT_CHUNKSIZE,
        low_memory=True,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="raise").astype("datetime64[ns]")
        chunk["ticker"] = chunk["ticker"].astype("category")
        chunk["return_mode"] = pd.Categorical(
            chunk["return_mode"], categories=_RETURN_MODE_CATEGORIES
        )
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols)
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    return frame


def _read_validation_analysis_and_metadata(
    path: Path, *, chunksize: int = DEFAULT_CHUNKSIZE
) -> tuple[pd.DataFrame, dict]:
    """Single-pass validation loader: audit metadata + five scientific columns.

    The ticker column exists only inside each bounded chunk, where it updates the
    universe audit set, and is dropped before the chunk is retained. This avoids
    a second full decompression pass over the 31.9M-row validation gzip file.
    """
    from .independent_validation import _canonical_json_sha256, _sha256_bytes

    path = Path(path)
    usecols = ["date", "ticker", "return_mode", "horizon", "psi_primary", "phase_product"]
    dtype = {
        "ticker": "string",
        "return_mode": "string",
        "horizon": "int16",
        "psi_primary": "float64",
        "phase_product": "float64",
    }
    parts: list[pd.DataFrame] = []
    tickers: set[str] = set()
    horizons: set[int] = set()
    modes: set[str] = set()
    rows = 0
    date_min = None
    date_max = None

    for idx, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            chunksize=int(chunksize),
            low_memory=True,
        ),
        start=1,
    ):
        rows += len(chunk)
        tickers.update(str(x) for x in chunk["ticker"].dropna().unique())
        horizons.update(int(x) for x in chunk["horizon"].dropna().unique())
        modes.update(str(x) for x in chunk["return_mode"].dropna().unique())

        chunk["date"] = pd.to_datetime(chunk["date"], errors="raise").astype("datetime64[ns]")
        if len(chunk):
            cmin = chunk["date"].min()
            cmax = chunk["date"].max()
            if pd.notna(cmin) and (date_min is None or cmin < date_min):
                date_min = cmin
            if pd.notna(cmax) and (date_max is None or cmax > date_max):
                date_max = cmax

        chunk["return_mode"] = pd.Categorical(
            chunk["return_mode"], categories=_RETURN_MODE_CATEGORIES
        )
        chunk["horizon"] = chunk["horizon"].astype("int16")
        chunk["psi_primary"] = chunk["psi_primary"].astype("float64")
        chunk["phase_product"] = chunk["phase_product"].astype("float64")
        chunk = chunk.loc[:, list(PANEL_ANALYSIS_COLUMNS)]
        parts.append(chunk)

        if idx % 20 == 0:
            print(f"stream-load validation: {rows:,} rows")

    if not parts:
        raise ValueError(f"Validation panel is empty: {path}")

    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    del parts
    gc.collect()

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

    mb = frame.memory_usage(index=True, deep=True).sum() / (1024.0 ** 2)
    print(
        f"stream-load validation-analysis: rows={len(frame):,} "
        f"cols={len(frame.columns)} ram={mb:,.1f} MiB"
    )
    return frame, meta


def _read_analysis_panel_streaming(path: Path, *, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """Backward-compatible scientific-only return value for tests/small callers."""
    frame, _ = _read_validation_analysis_and_metadata(path, chunksize=chunksize)
    return frame

def main() -> None:
    args = parse_args()
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))

    # Audit both files without ever materializing either full CSV as a DataFrame.
    discovery_meta = stream_panel_metadata(Path(args.discovery_panel), chunksize=DEFAULT_CHUNKSIZE)
    print(
        "stream-audit discovery: "
        f"rows={discovery_meta['rows']:,} tickers={discovery_meta['ticker_count']} "
        f"dates={discovery_meta['date_min']}..{discovery_meta['date_max']}"
    )
    # Validation audit + scientific extraction are combined in one streaming pass.
    validation, validation_meta = _read_validation_analysis_and_metadata(
        Path(args.validation_panel), chunksize=DEFAULT_CHUNKSIZE
    )
    print(
        "stream-audit validation: "
        f"rows={validation_meta['rows']:,} tickers={validation_meta['ticker_count']} "
        f"dates={validation_meta['date_min']}..{validation_meta['date_max']}"
    )

    result = run_independent_validation(
        discovery_panel=None,
        validation_panel=validation,
        protocol=protocol,
        out_dir=Path(args.out),
        independence_mode=args.independence,
        discovery_metadata=discovery_meta,
        validation_metadata=validation_meta,
    )
    print(json.dumps(result["assessment"], indent=2, default=str))


if __name__ == "__main__":
    main()

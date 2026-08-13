# Momentum–Reversal Spectral Phase Diagram (MRSPD)

[![CI](https://github.com/malikberrada/MomentumReversalSpectralPhaseDiagram/actions/workflows/ci.yml/badge.svg)](https://github.com/malikberrada/MomentumReversalSpectralPhaseDiagram/actions/workflows/ci.yml)

MRSPD is a research codebase for studying whether momentum and reversal are organized by an **horizon-dependent spectral geometry** rather than by one universal scalar threshold.

The repository preserves the full scientific logic used in the project: non-monotone phase-surface estimation, critical-horizon certification, independent-validation audits, percentile transport, robust upper-tail analysis, and the v10 cross-market confirmatory endpoint.

## Main scientific endpoint

The v10 endpoint asks whether, in a previously unseen national equity universe, the predefined upper spectral-percentile tail

\[
q_\Psi \ge 0.85
\]

enters a persistent reversal regime in both **raw** and **local-market-residual** returns, while allowing the critical onset \(h_c\) to be market-specific.

The recorded Japanese V4 confirmation is stored in `results/reference/v10_v4_japan_summary.json`:

- `design_audit_pass = true`
- `market_residual h_c = 96` trading sessions
- `raw h_c = 147` trading sessions
- consensus `h_c = 147`
- `overall_confirmatory_pass = true`
- frozen protocol SHA-256: `c57c1a21a7da9d7cbbb6ec2bcb93a08194874ee8de9899f06b4b351f0f6107f4`

Earlier non-replications are **not deleted**. The frozen v8 independent-validation failure is preserved under `results/reference/` to keep the research record auditable.

## Repository layout

```text
.
├── src/mrspd/                 # canonical Python implementation and CLIs
├── tests/                     # synthetic/unit/protocol tests (no market download)
├── data/universes/            # frozen ticker universes used by the research stages
├── data/cache/                # local yfinance cache (gitignored)
├── runs/                      # generated panels/statistical outputs (gitignored)
├── native/                    # optional C++/OpenMP/CUDA accelerator source
├── results/reference/         # compact reference verdicts and protocol hashes
├── reproducibility/           # hashes and reproducibility metadata
├── docs/                      # methodology and research-history notes
└── .github/workflows/ci.yml   # CPU CI on Linux + Windows
```

## Install

Python 3.10–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

The default scientific baseline is pure Python. The native backend is optional; see [`native/README.md`](native/README.md).

## Minimal reproducible smoke test

No market data are required:

```bash
python -m pytest -q
```

## Build a panel

The command downloads OHLCV from Yahoo Finance through `yfinance` and caches it locally.

Example discovery-style run:

```bash
mrspd \
  --universe data/universes/discovery_us_mixed.csv \
  --market-ticker SPY \
  --cache data/cache/ohlcv-discovery.pkl \
  --out runs/discovery \
  --start 2005-01-01 \
  --horizons "63,72,84,96,105,112,119,126,133,140,147,168" \
  --rebuild-panel \
  --native-backend off \
  --panel-only
```

For the exact V10 horizon family:

```text
63,72,84,96,105,112,119,126,133,140,147,168,189,210,231,252
```

## Japanese V4 workflow

The exact frozen V4 universe is already committed as `data/universes/confirmatory_japan_nikkei225.csv`. Do **not** regenerate it when reproducing the historical confirmation: current index membership can change.

A new prospective study may rebuild a fresh candidate list with:

```bash
mrspd-build-v4-nikkei225 --out data/universes/new_nikkei225.csv --expected-count 225
```

For a new prospective validation, freeze and bind the protocol **before** inspecting the new price panel. See [`docs/reproducibility.md`](docs/reproducibility.md).

## Reproducibility levels

**Code/protocol reproducibility:** this repository contains the canonical code, unit tests, frozen universe lists, reference protocol hash, and deterministic seeds.

**Bitwise data reproducibility:** Yahoo Finance is an external mutable data source. Exact byte-for-byte reconstruction of the historical panel requires the original `ohlcv*.pkl` cache, which is intentionally not redistributed here. Archive that cache separately with a checksum if the manuscript requires exact computational replication.

## Native acceleration

`native/` contains an optional compiled backend. The main code transparently falls back to pure Python if `mrspd_native` is absent.

CPU/OpenMP:

```bash
MRSPD_NATIVE_CUDA=0 python -m pip install ./native --no-build-isolation
```

CUDA auto mode:

```bash
python -m pip install ./native --no-build-isolation
```

The native backend is an accelerator, not a different statistical specification.

## Research integrity

The repository distinguishes exploratory development from frozen confirmatory testing. In particular:

- the v8 frozen independent validation did **not** replicate;
- the UK v9.2 confirmatory localization did **not** replicate under its frozen protocol;
- those failures motivated a **new** v10 hypothesis rather than retroactively changing the failed tests;
- v10 then tested a market-specific critical horizon on a separately frozen Japanese V4 universe.

See [`docs/scientific-history.md`](docs/scientific-history.md).

## Data

Only ticker universes are committed. Generated OHLCV caches and panels are ignored by Git.

| File | Role |
|---|---|
| `discovery_us_mixed.csv` | original mixed discovery universe |
| `validation_us_sp400_sp600.csv` | independent U.S. stock universe |
| `development_uk_ftse350.csv` | U.K. development/transport universe |
| `confirmatory_japan_nikkei225.csv` | frozen Japanese V4 universe |

## License

MIT. See [`LICENSE`](LICENSE).

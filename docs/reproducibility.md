# Reproducibility

## What is frozen in Git

- canonical source code;
- unit and protocol-invariant tests;
- ticker-universe CSVs used in the research stages;
- deterministic seed (`20260812`);
- compact reference verdicts;
- the final v10 protocol SHA-256 recorded by the confirmatory run.

## What is intentionally not stored

- Yahoo Finance OHLCV caches;
- multi-million-row `panel.csv.gz` files;
- generated figures and transient runs;
- compiler objects / Windows `.pyd` binaries;
- historical patch bundles and backups.

These artifacts made the development ZIP difficult to audit and are either generated, platform-specific, or redundant with the canonical code.

## Exact historical replication

Yahoo Finance can revise adjusted histories and corporate-action metadata. Therefore a future download may be scientifically equivalent without being byte-identical to the 2026 run.

For archival-grade replication, preserve the original cache outside Git and record:

```bash
sha256sum data/cache/ohlcv-*.pkl
sha256sum runs/*/panel.csv.gz
```

Then publish the immutable archive through a suitable data repository if licensing permits.

## Prospective validation rule

When testing a new universe:

1. define the candidate universe without reading its outcome panel;
2. bind its exact ticker set and local market benchmark;
3. freeze the protocol and record its SHA-256;
4. only then build/download the price panel;
5. run the confirmatory validator without changing thresholds, horizons, fold rules, bootstrap settings, or persistence rules.

## CPU versus native backends

The pure-Python implementation is the baseline. Native C++/CUDA is optional acceleration. Unit tests should be run without requiring native compilation; native-equivalence checks may be run separately on compatible hardware.

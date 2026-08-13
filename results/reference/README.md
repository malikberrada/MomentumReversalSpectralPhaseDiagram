# Reference verdicts

These compact JSON files preserve the **reported scientific verdicts** needed to understand the development sequence. They are not substitutes for the full generated run directories.

- `v8_independent_validation_forensic_summary.json` preserves the failed frozen v8 independent-validation verdict and the fact that implementation invariants passed.
- `v10_v4_japan_summary.json` preserves the recorded frozen v10 Japanese confirmatory verdict and its protocol SHA-256.

For archival-grade replication, also preserve the corresponding generated protocol JSON, manifest, OHLCV cache, panel, and full confirmatory output directory outside Git. The development ZIP supplied to build this clean repository did not contain those multi-million-row runtime artifacts.

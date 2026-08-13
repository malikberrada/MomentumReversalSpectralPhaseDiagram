# Contributing

Please keep scientific changes separate from performance/refactoring changes. Any change to thresholds, horizon grids, bootstrap rules, fold support, multiplicity correction, percentile-tail definition, or validation endpoints must be documented as a new protocol version rather than silently modifying an existing frozen protocol.

Before opening a pull request:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Do not commit downloaded OHLCV caches, generated panels, compiled native binaries, or development backups.

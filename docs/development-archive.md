# Development archive

The source ZIP contained many patch bundles, PowerShell apply scripts, backups, duplicate payload trees, `.pyc` files, and compiled Windows build objects created during v4–v10 development.

They are **not** copied into the clean GitHub tree because:

1. they duplicate the canonical final modules under `src/mrspd/`;
2. old patch payloads can accidentally shadow current tests/modules;
3. compiled objects are platform-specific and non-reproducible source artifacts;
4. backups make it ambiguous which implementation is authoritative.

The original development ZIP should be retained separately as provenance, but not used as the public code entry point.

# Methodology overview

The implementation defines a continuous phase product and studies its conditional expectation over spectral state and horizon. Positive conditional values correspond to persistence/momentum; negative conditional values correspond to reversal.

The repository contains:

- spectral feature construction (`pipeline.py`);
- non-monotone spline phase surfaces (`phase_surface.py`);
- OOS root/single-phase certification with moving-block bootstrap;
- coarse and refined critical-horizon scans;
- frozen independent-validation protocols and audits;
- percentile transport across universes;
- robust Huber-clipped upper-tail response;
- market-specific critical-horizon confirmation in v10.

The main inferential guardrails are walk-forward splits, block bootstrap for temporally dependent observations, multiplicity correction, minimum effect floors, fold-support requirements, and frozen-protocol SHA-256 validation.

For exact formulae and implementation details, read the module docstrings/source and the generated protocol JSONs produced by the freeze commands.

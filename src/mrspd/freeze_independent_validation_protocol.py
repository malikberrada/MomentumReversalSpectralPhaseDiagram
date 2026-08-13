from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .independent_validation import freeze_validation_protocol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze MRSPD h_c independent-validation protocol")
    p.add_argument("--discovery-panel", required=True)
    p.add_argument("--coarse-summary", required=True)
    p.add_argument("--refinement-summary", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(Path(args.discovery_panel), parse_dates=["date"])
    coarse = json.loads(Path(args.coarse_summary).read_text(encoding="utf-8"))
    refine = json.loads(Path(args.refinement_summary).read_text(encoding="utf-8"))
    protocol = freeze_validation_protocol(
        discovery_panel=panel,
        discovery_coarse_summary=coarse,
        discovery_refinement_summary=refine,
        out_path=Path(args.out),
    )
    print(json.dumps(protocol, indent=2, default=str))


if __name__ == "__main__":
    main()

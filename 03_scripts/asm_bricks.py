"""Compatibility entry point for preprocessing a CRAG ground-truth GLB.

The generalized pipeline intentionally does not consume predicted-pose GLBs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assembly_pipeline import main as pipeline_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pred", help="unsupported in the generalized ground-truth pipeline")
    parser.add_argument("--length", type=float, default=0.40)
    parser.add_argument("--pitch", type=float, default=0.016)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stable", action="store_true", help="retained for CLI compatibility")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.pred:
        parser.error("--pred is outside the generalized ground-truth pipeline")
    command = [
        "prepare",
        "--dataset",
        "crag-glb",
        "--input",
        str(arguments.glb),
        "--output",
        str(arguments.out),
        "--target-length",
        str(arguments.length),
        "--pitch",
        str(arguments.pitch),
        "--seed",
        str(arguments.seed),
    ]
    if arguments.overwrite:
        command.append("--overwrite")
    return pipeline_main(command)


if __name__ == "__main__":
    raise SystemExit(main())

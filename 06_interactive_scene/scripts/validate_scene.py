"""Headless validation entry point for the interactive scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from build_scene import build_stage, validate_manifest
from scene_config import SceneConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.manifest_only:
        parser.error("this build currently supports --manifest-only")
    config = SceneConfig.load(arguments.config)
    handles = build_stage(config, headless=True)
    try:
        report = validate_manifest(handles)
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "fragment_count": report.fragment_count,
                    "missing_paths": report.missing_paths,
                    "extra_fragment_paths": report.extra_fragment_paths,
                },
                indent=2,
            )
        )
        return 0 if report.ok else 1
    finally:
        handles.app.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run all pipeline tests and generate visual reports.

Usage:
    python run_visuals.py                  # run all tests, overwrite existing visuals
    python run_visuals.py --no-overwrite   # skip tests whose output images already exist
    python run_visuals.py --stage 1 2      # run only stages 1 and 2
    python run_visuals.py --visual-only    # run only visual tests (skip correctness)
    python run_visuals.py --correct-only   # run only correctness tests (skip visuals)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "tests" / "visual_reports"

STAGE_FILES = {
    1: "tests/test_stage1_cleanup.py",
    2: "tests/test_stage2_segment.py",
    3: "tests/test_stage3_fit.py",
    4: "tests/test_stage4_topology.py",
    5: "tests/test_stage5_features.py",
    6: "tests/test_stage6_export.py",
}

STAGE_FOLDERS = {
    1: "stage1_cleanup",
    2: "stage2_segment",
    3: "stage3_fit",
    4: "stage4_topology",
    5: "stage5_features",
    6: "stage6_export",
}


def main():
    parser = argparse.ArgumentParser(description="Run mesh2param tests and generate visual reports.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip visual tests whose output images already exist.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5, 6],
        help="Run only the specified stage(s). Default: all.",
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Run only visual (screenshot-generating) tests.",
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Run only correctness tests (no visual output).",
    )
    args = parser.parse_args()

    stages = args.stage or [1, 2, 3, 4, 5, 6]

    # Clear old visuals for selected stages (unless --no-overwrite)
    if not args.no_overwrite and not args.correct_only:
        for s in stages:
            folder = REPORTS_DIR / STAGE_FOLDERS[s]
            if folder.exists():
                shutil.rmtree(folder)
                print(f"  Cleared {folder}")

    # Build pytest args
    test_files = [STAGE_FILES[s] for s in stages]
    pytest_args = [sys.executable, "-m", "pytest"] + test_files + ["-v", "--tb=short"]

    if args.visual_only:
        pytest_args += ["-m", "visual"]
    elif args.correct_only:
        pytest_args += ["-m", "not visual"]

    print(f"\n{'='*60}")
    print(f"  Running stages: {stages}")
    print(f"  Overwrite: {not args.no_overwrite}")
    print(f"  Output: {REPORTS_DIR}")
    print(f"{'='*60}\n")

    result = subprocess.run(pytest_args)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Visual reports: {REPORTS_DIR}")
    for s in stages:
        folder = REPORTS_DIR / STAGE_FOLDERS[s]
        if folder.exists():
            pngs = list(folder.glob("*.png"))
            print(f"  Stage {s}: {len(pngs)} images in {folder.name}/")
        else:
            print(f"  Stage {s}: no output (pipeline not implemented)")
    print(f"{'='*60}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

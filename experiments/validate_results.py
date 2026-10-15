"""CLI: check a finished experiment's results/<name>/ for signs of
corruption or a broken pipeline, and write a validation_report.json marker
so a human glancing at the directory immediately sees whether it's
trustworthy - the "don't let us read gibberish and think it's real" gate,
run automatically at the end of experiments/run.py::main() for a fresh run,
or standalone here for any existing results directory.

Usage:
    uv run experiments/validate_results.py --results-dir results/no_mitigation
    uv run experiments/validate_results.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurofault.validation.report import validate_results_dir
from neurofault.validation.results_validator import Level


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", type=Path)
    group.add_argument("--all", action="store_true", help="validate every subdirectory of results/")
    args = parser.parse_args()

    if args.all:
        results_dirs = sorted(p for p in Path("results").iterdir() if p.is_dir())
    else:
        results_dirs = [args.results_dir]

    worst = Level.PASS
    for results_dir in results_dirs:
        verdict = validate_results_dir(results_dir)
        if verdict == Level.FAIL:
            worst = Level.FAIL
        elif verdict == Level.WARN and worst != Level.FAIL:
            worst = Level.WARN

    sys.exit(1 if worst == Level.FAIL else 0)


if __name__ == "__main__":
    main()

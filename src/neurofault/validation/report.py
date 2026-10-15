"""Loads a finished run's metrics.json/config.json, runs
results_validator.validate() against them, prints a human-readable summary,
and writes validation_report.json as the on-disk marker.

Lives under the installed `neurofault` package (not `experiments/`) so it's
reliably importable both from `experiments/run.py` (invoked as a bare script,
`uv run experiments/run.py ...` - the project's own established invocation
style throughout during development) and from `experiments/validate_results.py`'s
CLI. `experiments` itself is only guaranteed importable as a package when
run via its registered console script (`neurofault-run`), not as a bare
script path - verified directly: `from experiments.validate_results import
...` inside experiments/run.py raised `ModuleNotFoundError: No module named
'experiments'` under the bare-script invocation this project actually uses.
"""

from __future__ import annotations

import json
from pathlib import Path

from neurofault.validation.results_validator import Level, overall_verdict, validate


def validate_results_dir(results_dir: Path) -> Level:
    metrics = json.loads((results_dir / "metrics.json").read_text())
    config = json.loads((results_dir / "config.json").read_text())

    events_path = results_dir / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else None

    results = validate(metrics, config, results_name=results_dir.name, events=events)
    verdict = overall_verdict(results)

    report = {
        "results_dir": str(results_dir),
        "overall_verdict": verdict.value,
        "checks": [{"name": r.name, "level": r.level.value, "message": r.message} for r in results],
    }
    (results_dir / "validation_report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{results_dir}: {verdict.value}")
    for r in results:
        if r.level != Level.PASS:
            print(f"  [{r.level.value}] {r.name}: {r.message}")

    return verdict

"""F0.0 / F0.1 smoke variants run end to end (< 60 s each) and write verdict.json.

Smoke results never count; this only checks the plumbing: scripts run, emit the metric columns
the frozen criteria reference, and produce a verdict file with the expected criteria."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from neurofault.spectral.tiles import checkpoint_path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiments" / "spectral"

pytestmark = pytest.mark.skipif(
    not checkpoint_path(42).exists(), reason="ResNet-9 checkpoint results/_live2 not present"
)


def _run(script: str, out: Path) -> dict:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--smoke", "--allow-dirty", "--jobs", "2",
         "--out", str(out)],
        capture_output=True, text=True, timeout=120, check=False, cwd=ROOT,
        env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )  # fmt: skip
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = [d for d in out.iterdir() if d.is_dir() and (d / "verdict.json").exists()]
    assert len(runs) == 1
    assert (runs[0] / "metrics.csv").exists() and (runs[0] / "run_metadata.json").exists()
    return json.loads((runs[0] / "verdict.json").read_text())


@pytest.mark.skipif(
    importlib.util.find_spec("aihwkit") is None,
    reason="AIHWKit not installed (F0.0b compares against its noise models)",
)
def test_f0_0_smoke(tmp_path):
    v = _run("f0_0_validity.py", tmp_path / "f0_0")
    st = {c["id"]: c["status"] for c in v["criteria"]}
    assert {"F0.0a", "F0.0b", "F0.0c", "F0.0d", "F0.0e"} <= set(st)
    for cid in ("F0.0c", "F0.0d", "F0.0e"):  # deterministic gates must pass even in smoke
        assert st[cid] == "PASS", st
    assert st["F0.8a"] == "INCONCLUSIVE"  # bridge gates are out of scope for this script


def test_f0_1_smoke(tmp_path):
    v = _run("f0_1_stuck_scaling.py", tmp_path / "f0_1")
    ids = {c["id"] for c in v["criteria"]}
    assert {"H1.1", "H1.2", "H1.3-line", "H1.3-kill", "H1.4", "H1.6"} <= ids
    # every metric the rules reference must exist as a column (no 'missing' no-data clauses)
    for c in v["criteria"]:
        for leaf in _leaves(c["detail"]):
            assert "missing" not in str(leaf.get("reason")), (c["id"], leaf)


def _leaves(d):
    if "clauses" in d:
        for c in d["clauses"]:
            yield from _leaves(c)
    elif "metric" in d:
        yield d

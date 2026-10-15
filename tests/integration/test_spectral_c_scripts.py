"""C' smoke variant (C-X1) runs end to end and feeds the frozen criteria.

Smoke results never count; this checks plumbing: the script runs and emits the metric columns
the frozen criteria file references (no 'missing' leaves for the criteria it feeds)."""

from __future__ import annotations

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


def _run(script: str, out: Path, timeout: int = 240) -> tuple[dict, Path]:
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--smoke", "--allow-dirty", "--jobs", "1",
         "--out", str(out)],
        capture_output=True, text=True, timeout=timeout, check=False, cwd=ROOT,
        env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )  # fmt: skip
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = [d for d in out.iterdir() if d.is_dir() and (d / "verdict.json").exists()]
    assert len(runs) == 1
    return json.loads((runs[0] / "verdict.json").read_text()), runs[0]


def _leaves(d):
    if "clauses" in d:
        for c in d["clauses"]:
            yield from _leaves(c)
    elif "metric" in d:
        yield d


def _status(v):
    return {c["id"]: c["status"] for c in v["criteria"]}


def test_c_x1_smoke_feeds_s1_k1_d1(tmp_path):
    v, run_dir = _run("c_x1.py", tmp_path / "c_x1")
    fed = {"S1", "K1a", "K1b", "K1c", "D1", "KX1"}
    for c in v["criteria"]:
        if c["id"] in fed:
            for leaf in _leaves(c["detail"]):
                assert "missing" not in str(leaf.get("reason")), (c["id"], leaf)
    assert (run_dir / "gap_variation.csv").exists()

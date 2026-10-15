"""Every shipped spectral verdict was scored against the criteria file that ships with it.

`neurofault.spectral.verdict` records the sha256 of the criteria YAML it loaded. The criteria
files are frozen pre-registrations, so they must stay byte-identical: editing one (even a
comment) breaks the link between a published verdict and the rules it was judged by.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT / "experiments" / "spectral" / "criteria"
RESULTS = ROOT / "results" / "spectral"


def _verdicts() -> list[Path]:
    return sorted(RESULTS.rglob("verdict*.json")) if RESULTS.is_dir() else []


def test_every_verdict_matches_a_shipped_criteria_file() -> None:
    verdicts = _verdicts()
    if not verdicts:
        pytest.skip("no shipped spectral verdicts")
    shipped = {hashlib.sha256(p.read_bytes()).hexdigest(): p.name for p in CRITERIA.glob("*.yaml")}
    checked, missing = 0, []
    for path in verdicts:
        data = json.loads(path.read_text())
        digest = data.get("criteria_sha256") if isinstance(data, dict) else None
        if not digest:
            continue
        checked += 1
        if digest not in shipped:
            missing.append(str(path.relative_to(ROOT)))
    assert checked > 0
    assert not missing, (
        f"verdicts scored against criteria that do not ship byte-identical: {missing}"
    )

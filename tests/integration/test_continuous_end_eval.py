"""Continuous policy runs record end-of-training test accuracy in events.json summary."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import run_experiment
from tests.integration.test_continuous_policy import _policy_config


def test_summary_has_end_of_run_test_acc(tmp_path):
    run_experiment(_policy_config("capacity", remap_order="random"), tmp_path)
    s = json.loads((tmp_path / "events.json").read_text())["summary"]
    assert isinstance(s["end_of_run_test_acc"], float)
    assert 0.0 <= s["end_of_run_test_acc"] <= 100.0
    assert "final_test_acc" in s  # unchanged: value right after the last wave

"""Byte-equal regression for the run.py training-loop refactor.

tests/golden/refactor_golden_* were generated from the committed pre-refactor code
(see tests/golden/README.md). The config disables remapping so unrelated remap/dispatch
fixes cannot move it; soft mitigation, layer reset and a float-pretrain phase are all
exercised. The continuous path must reproduce it exactly - same loss/accuracy/faults/
health/mitigation counters/grad_norm and the same per-layer health snapshots - which
also proves the refactor consumed no extra (or reordered) RNG.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

from experiments.run import run_experiment
from neurofault.config import load_config

GOLDEN = Path(__file__).resolve().parent.parent / "golden"


@pytest.mark.slow  # the real ResNet-9 through 9 crossbars: the golden is the point
def test_continuous_path_is_byte_equal_to_pre_refactor_golden():
    config = load_config(GOLDEN / "refactor_golden_config.yaml")
    metrics = run_experiment(config)
    layer_health = metrics.pop("layer_health")
    metrics.pop("epoch_seconds")  # wall-clock, the only non-deterministic key

    assert json.dumps(metrics, indent=2) == (GOLDEN / "refactor_golden_metrics.json").read_text()
    assert (
        json.dumps(layer_health, indent=2)
        == (GOLDEN / "refactor_golden_layer_health.json").read_text()
    )
    # The golden is only a meaningful guard if the paths it targets actually fired.
    assert metrics["soft_mitigations"][-1] > 0
    assert metrics["layer_resets"][-1] > 0
    assert metrics["faults"][-1] > 0

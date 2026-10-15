"""A real (if tiny) run through experiments/run.py's actual CLI entrypoint
(main(), not run_experiment() directly - this is specifically testing what
main() writes to disk), asserting every artifact the development-time
logging/provenance work added actually gets produced and is well-formed:
run_metadata.json (git+environment+timing), run.log (raw stdout capture),
layer_health.json (the per-crossbar detail previously discarded down to one
avg_health scalar), and metrics.json's new grad_norm/epoch_seconds fields.
Marked slow - a real training run, not part of the fast default loop.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time

import pytest
import yaml

pytest.importorskip("simulator", reason="CrossSim not installed (uv sync --extra crosssim)")

import experiments.run as run_module
from experiments.run import main, run_experiment

pytestmark = pytest.mark.slow


def _write_tiny_config(path, output_dir, save_checkpoint=False) -> None:
    config = {
        "name": "test_run_output_artifacts",
        "simulator": "crosssim",
        "seed": 7,
        "epochs": 2,
        "num_batches": 4,
        "train_samples": 32,
        "test_samples": 16,
        "batch_size": 8,
        "eval_interval": 1,
        "learning_rate": 0.01,
        "output_dir": str(output_dir),
        "save_checkpoint": save_checkpoint,
        "crossbar": {"patch_layer_types": ["Linear"]},
        "fault": {"density": 0.0},
    }
    path.write_text(yaml.safe_dump(config))


def test_main_writes_every_expected_artifact(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "results"
    _write_tiny_config(config_path, output_root)

    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    main()

    run_dir = output_root / "test_run_output_artifacts"
    for filename in (
        "metrics.json",
        "config.json",
        "run_metadata.json",
        "run.log",
        "layer_health.json",
        "plots.png",
        "validation_report.json",
    ):
        assert (run_dir / filename).exists(), f"{filename} was not written"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "layer_health" not in metrics, "layer_health must be popped into its own file"
    assert len(metrics["grad_norm"]) == len(metrics["loss"])
    assert all(math.isfinite(v) for v in metrics["grad_norm"])
    assert len(metrics["epoch_seconds"]) == 2  # one per epoch, not per eval point
    assert all(v > 0 for v in metrics["epoch_seconds"])

    layer_health = json.loads((run_dir / "layer_health.json").read_text())
    assert isinstance(layer_health, list)
    assert len(layer_health) == len(metrics["loss"])
    assert "layers" in layer_health[0]

    run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert run_metadata["git"]["commit"] is not None
    assert isinstance(run_metadata["git"]["dirty"], bool)
    assert run_metadata["environment"]["torch_version"]
    assert run_metadata["timing"]["total_seconds"] > 0

    assert (run_dir / "run.log").read_text().strip() != ""
    assert not (run_dir / "checkpoint.pt").exists()  # save_checkpoint=False


def test_save_checkpoint_writes_checkpoint_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "results"
    _write_tiny_config(config_path, output_root, save_checkpoint=True)

    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
    main()

    run_dir = output_root / "test_run_output_artifacts"
    assert (run_dir / "checkpoint.pt").exists()


def test_metrics_are_written_incrementally_not_only_at_the_end(tmp_path, monkeypatch):
    """Real gap closed 2026-09-29: a rented GPU
    instance destroyed mid-run previously took the whole run's progress with it, since
    metrics.json/layer_health.json were only ever written once, after run_experiment()
    returned. Verifies the fix directly: runs a real (if tiny) training loop on a
    background thread with the snapshot interval forced very low, and confirms
    metrics.json genuinely changes content WHILE the run is still in progress - not just
    once the thread has already finished."""
    from neurofault.config import ExperimentConfig

    monkeypatch.setattr(run_module, "_SNAPSHOT_INTERVAL_SECONDS", 0.0)

    output_dir = tmp_path / "run_output"
    output_dir.mkdir()
    config = ExperimentConfig(
        name="incremental_probe",
        simulator="crosssim",
        seed=7,
        epochs=1,
        num_batches=30,
        train_samples=240,
        test_samples=16,
        batch_size=8,
        eval_interval=1,
        learning_rate=0.01,
    )
    config.crossbar.patch_layer_types = ["Linear"]
    config.fault.density = 0.0

    thread = threading.Thread(target=run_experiment, args=(config, output_dir))
    thread.start()

    observed_lengths = set()
    metrics_path = output_dir / "metrics.json"
    deadline = time.time() + 30
    while thread.is_alive() and time.time() < deadline:
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text())
                observed_lengths.add(len(data["loss"]))
            except (json.JSONDecodeError, KeyError):
                pass  # caught mid-write despite the atomic rename - vanishingly rare, just retry
        time.sleep(0.05)

    thread.join(timeout=30)
    assert not thread.is_alive(), "run_experiment did not finish in time"

    # The real assertion: metrics.json was observed with at least 2 DIFFERENT lengths
    # while the background thread was still running - proof it was updated incrementally,
    # not written once at the very end (which would show at most 1 length: 0, or the
    # thread would already be dead by the time anything was observed).
    assert len(observed_lengths) >= 2, (
        f"expected metrics.json to grow incrementally during the run, only observed "
        f"{observed_lengths} - looks like it's still only written once at the end"
    )

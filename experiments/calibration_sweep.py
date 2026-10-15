"""GPU calibration sweep for the co-mitigation controller (calibration step 2).

For each (fault fraction, burst learning rate eta) this REBUILDS the crossbar system
fresh from the pre-deploy checkpoint, injects ONE wave at that fraction with the same
dedicated generator run_deployment uses for wave 0 (seed*1000+0) so the fault map is
identical for every eta at a given fraction, does NO remap (the curve measures raw
retraining recovery from u = fraction unremapped stuck cells; the stuck-weight freeze is
on, as in the deployment regime), then runs a fixed-length burst at that eta with a
CANARY eval every `--eval-every` batches. Only canary accuracy is recorded - never the
test set - so thresholds later tuned on these curves cannot leak test accuracy.

Output: <out>/canary_curves.json
  {"metadata": {...},
   "curves": {"<fraction>": {"<eta>": [[batch, canary_acc], ...]}}}
with batch 0 = the canary right after the wave (before any step). Consumed by
scripts/calibrate_retrain_controller.py --from-calibration-sweep.

Usage:
    uv run experiments/calibration_sweep.py --config experiments/configs/resnet9_calibration_sweep.yaml \\
        --checkpoint results/resnet9_predeploy/pre_deploy.pt --out results/calibration_sweep
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see experiments/run.py

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in (None, ""):  # bare `uv run experiments/calibration_sweep.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.run import _evaluate_model, _inject_wave, _load_deployment_system
from neurofault.config import ExperimentConfig, load_config
from neurofault.provenance import capture_git_info
from neurofault.retraining.burst import run_burst
from neurofault.retraining.canary import build_canary

logger = logging.getLogger(__name__)

DEFAULT_PLAN = (
    ([0.10], [1e-4, 1e-3, 5e-3]),
    ([0.05, 0.20], [1e-3]),
)


def default_runs() -> list[tuple[float, float]]:
    return [(f, e) for fractions, etas in DEFAULT_PLAN for f in fractions for e in etas]


def run_one(
    config: ExperimentConfig, fraction: float, eta: float, batches: int, eval_every: int
) -> tuple[list[list[float]], float]:
    """One fresh-system run. Returns ([[batch, canary_acc], ...] incl. batch 0, the
    pre-wave clean canary accuracy)."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if config.crossbar.use_gpu else "cpu")
    system, _, fisher, _, _ = _load_deployment_system(config, device)
    canary = build_canary(config)  # fresh burst-shuffle generator: same batch order per run
    system.model.eval()
    clean = _evaluate_model(system.model, canary.canary_loader, device)

    base_fault = config.fault
    system.config = replace(system.config, fault=replace(base_fault, density=fraction))
    _inject_wave(system, config.seed * 1000 + 0)
    system.apply_pins()
    post_wave = _evaluate_model(system.model, canary.canary_loader, device)

    result = run_burst(
        system,
        canary.burst_loader,
        lr=eta,
        n_batches=batches,
        canary=canary.canary_loader,
        eval_every=eval_every,
        fisher=fisher,
        config=config,
        device=device,
    )
    return [[0, post_wave], *result.canary_trace], clean


def _free_gpu() -> None:
    """Each run_one rebuilds a full crossbar system; without this the cupy pool and torch cache
    keep ~2.5GB per run and an 8-run sweep OOMs a 16GB card (seen 2026-10-06, 6th run)."""
    import gc

    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001, S110 - cupy is optional (CPU smoke runs)
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(
    config: ExperimentConfig,
    out_dir: Path,
    runs: list[tuple[float, float]],
    batches: int,
    eval_every: int,
) -> dict:
    curves: dict[str, dict[str, list]] = {}
    clean_accs: list[float] = []
    for fraction, eta in runs:
        t0 = time.time()
        curve, clean = run_one(config, fraction, eta, batches, eval_every)
        _free_gpu()
        clean_accs.append(clean)
        curves.setdefault(repr(float(fraction)), {})[repr(float(eta))] = curve
        logger.info(
            "[sweep] fraction=%g eta=%g canary clean=%.2f post-wave=%.2f final=%.2f (%.0fs)",
            fraction,
            eta,
            clean,
            curve[0][1],
            curve[-1][1],
            time.time() - t0,
        )
    ckpt = torch.load(config.pre_deploy_checkpoint, map_location="cpu", weights_only=False)
    data = {
        "metadata": {
            # clean_acc is the pre-wave CANARY accuracy (the simulator's reference level);
            # identical across runs because the system is rebuilt from the same checkpoint.
            "clean_acc": clean_accs[0],
            "n_canary": config.retraining.canary_size,
            "seed": config.seed,
            "checkpoint": str(config.pre_deploy_checkpoint),
            "checkpoint_config_hash": ckpt.get("config_hash"),
            "batches": batches,
            "eval_every": eval_every,
            "wave_seed": config.seed * 1000,
            "remap": False,
            "git": capture_git_info(),
        },
        "curves": curves,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canary_curves.json").write_text(json.dumps(data, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="a deployment config (model/data settings)")
    ap.add_argument("--checkpoint", required=True, help="pre_deploy.pt")
    ap.add_argument("--etas", type=float, nargs="+")
    ap.add_argument(
        "--add-etas",
        type=float,
        nargs="+",
        help="extra etas appended to the reference-fraction (0.10) group of the DEFAULT plan "
        "(other fractions unchanged); not combinable with --etas/--fault-fractions",
    )
    ap.add_argument("--fault-fractions", type=float, nargs="+")
    ap.add_argument("--batches", type=int, default=780)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if (args.etas is None) != (args.fault_fractions is None):
        ap.error("--etas and --fault-fractions go together (default: the default 5-run grid)")
    if args.add_etas and args.etas is not None:
        ap.error("--add-etas extends the default plan; do not combine with --etas")
    if args.batches < 1 or args.eval_every < 1:
        ap.error("--batches and --eval-every must be >= 1")

    config = replace(load_config(args.config), pre_deploy_checkpoint=args.checkpoint)
    runs = default_runs()
    if args.add_etas:
        have = {e for f, e in runs if f == 0.10}
        runs = (
            [r for r in runs if r[0] == 0.10]
            + [(0.10, e) for e in args.add_etas if e not in have]
            + [r for r in runs if r[0] != 0.10]
        )
    runs = runs if args.etas is None else [(f, e) for f in args.fault_fractions for e in args.etas]
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run_sweep(config, Path(args.out), runs, args.batches, args.eval_every)
    logger.info("wrote %s", Path(args.out) / "canary_curves.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

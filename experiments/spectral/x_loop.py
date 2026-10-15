"""Closed loop X.1/X.2 (criteria/x_loop.yaml): run an unmodified deployment config (e.g. the co-mitigation
controller D) with the controller's fault map coming from counted probe reads instead of the
injector's truth.

Wraps experiments/run.py: identical CLI, plus
  --locator {oracle, adaptive, readout}   oracle = the unmodified path (no estimator attached)
  --sigma   read-noise sd as a fraction of the tile max|w| (default 0.02)
  --no-shadow  skip the shadow locator (cost/accuracy logging of the other locator)

  python experiments/spectral/x_loop.py --locator adaptive --config <cfg.yaml> --seed 42 \
      --pre-deploy-checkpoint results/resnet9_predeploy_seed42/pre_deploy.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import run

from neurofault.spectral.closed_loop import ProbeEstimator


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--locator", choices=["oracle", "adaptive", "readout"], required=True)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--no-shadow", action="store_true")
    # CS probes per tile: the H8 calibration-selected forward config (P24 at sigma <= 2%, P32 at 4%)
    ap.add_argument("--p-cs", type=int, default=None)
    a, rest = ap.parse_known_args()
    p_cs = a.p_cs or (24 if a.sigma <= 0.02 else 32)

    load_config = run.load_config
    tag = a.locator if a.locator == "oracle" else f"{a.locator}_s{a.sigma:g}"

    def load_tagged(path):  # distinct results/<name>/ per locator
        cfg = load_config(path)
        return run.replace(cfg, name=f"{cfg.name}_x{tag}")

    run.load_config = load_tagged
    if a.locator != "oracle":
        load = run._load_deployment_system

        def load_estimated(config, device):
            system, *other = load(config, device)
            system.fault_map_source = "estimated"
            system.fault_estimator = ProbeEstimator(
                kind=a.locator,
                sigma_frac=a.sigma,
                seed=config.seed,
                shadow=not a.no_shadow,
                p_cs=p_cs,
            )
            return (system, *other)

        run._load_deployment_system = load_estimated
    sys.argv = [sys.argv[0], *rest]
    run.main()


if __name__ == "__main__":
    main()

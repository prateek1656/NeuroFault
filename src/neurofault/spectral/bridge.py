"""Network-bridge tiers 2 and 3 and the F0.8 metric helpers (F0.8).

Tier 1 is the weight-space digital twin (experiments/spectral/a_common.NetBridge /
a_exp2_jobs.Exp2Bridge): effective conv matrices are written into the float model.
Tier 2 (`TierTwo`) pushes the SAME matrices (and the same mean-bias corrections) into the
repo's CrossSim-backed system through `accessor.write` and evaluates through it, so the
comparison isolates what CrossSim adds (8-bit ADC/DAC, its weight-range mapping), nothing else.
Tier 3 (`aihwkit_native`) is AIHWKit's own PCM drift + GlobalDriftCompensation on the same
network.

M5.0 probe findings this module relies on (probed 2026-10-10, CrossSim in this repo's config):
- `accessor.read()` has the orientation `weight.flatten(1)` on all nine real ResNet-9 layers
  (max |diff| 6e-8, float32 rounding).
- `set_matrix` adds NO error: programming_error / drift_error / read_noise are disabled, the
  read-back is exact to float32 (5.6e-9), repeated reads and repeated forwards are bit-identical.
- `set_matrix` re-derives core.min/core.max from the written matrix (no clipping of
  compensated weights that grow beyond the original range).
- The only tier-2 nonidealities are therefore the 8-bit DAC and the calibrated 8-bit ADC.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path

import torch

DEFAULT_CROSSSIM_CONFIG = "experiments/configs/resnet9_deploy_a.yaml"


def binom_halfwidth_pt(acc_pct: float, n: int, z: float = 1.96) -> float:
    """Half-width (in accuracy points) of the normal-approximation binomial CI of an accuracy
    measured on n test images. Uses the observed accuracy (clamped away from 0/100)."""
    p = min(max(acc_pct / 100.0, 1e-4), 1 - 1e-4)
    return 100.0 * z * math.sqrt(p * (1.0 - p) / n)


def abs_dev_over_halfwidth(acc_pct: float, stored_pct: float, n: int) -> float:
    """|acc - stored| in units of the binomial CI half-width at the stored accuracy (F0.8c)."""
    return abs(acc_pct - stored_pct) / binom_halfwidth_pt(stored_pct, n)


def f08_metric_rows(
    t12: list[dict], t3: list[dict], nodrift: list[dict], n_eval: int
) -> list[dict]:
    """Rows carrying exactly the frozen F0.8 metric names (criteria/f0_validity.yaml).

    t12:     dicts with acc_t1, acc_t2 (percent) per matched drifted cell -> f08a
    t3:      dicts with acc_ours, acc_aihwkit (percent) per (seed, time)  -> f08b
    nodrift: dicts with acc_t1, stored (percent)                           -> f08c
    """
    out = []
    for r in t12:
        out.append({**r, "f08a_tier1_vs_tier2_abs_diff_pt": abs(r["acc_t1"] - r["acc_t2"])})
    for r in t3:
        out.append({**r, "f08b_ours_vs_aihwkit_abs_diff_pt": abs(r["acc_ours"] - r["acc_aihwkit"])})
    for r in nodrift:
        out.append(
            {**r, "f08c_abs_dev_over_binomial_halfwidth": abs_dev_over_halfwidth(
                r["acc_t1"], r["stored"], n_eval)}
        )  # fmt: skip
    return out


def layer_of_handle(handle_name: str) -> str:
    """'res1.conv_a_crossbar_0' -> 'res1.conv_a.weight' (crosssim_backend.build_handles naming)."""
    return handle_name.removesuffix("_crossbar_0") + ".weight"


class TierTwo:
    """CrossSim-backed system of one network, evaluated with externally supplied matrices.

    `base_model` is deep-copied (patching is in place) and is the model whose weights the ADC
    is calibrated on (320 train images, as run_deployment does). `bn_of` maps conv name
    (no suffix) to its BatchNorm name; a mean-bias correction b is applied the way the digital
    twin does: running_mean -= b on that BatchNorm."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        bn_of: Mapping[str, str],
        seed: int,
        config_path: str | Path = DEFAULT_CROSSSIM_CONFIG,
        adc_bits: int | None = None,
        dac_bits: int | None = None,
        n_cal: int = 320,
        repo_root: Path | None = None,
    ):
        from dataclasses import replace

        from neurofault.config import load_config
        from neurofault.data import load_datasets
        from neurofault.system import FaultTolerantNeuromorphic

        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        cfg = load_config(root / config_path)
        xb = replace(cfg.crossbar, use_gpu=False)
        if adc_bits is not None:
            xb = replace(xb, adc_resolution=adc_bits)
        if dac_bits is not None:
            xb = replace(xb, dac_resolution=dac_bits)
        cfg = replace(cfg, seed=seed, crossbar=xb)
        torch.manual_seed(seed)  # the calibration loader shuffles
        trainloader, _ = load_datasets(cfg)
        imgs, n = [], 0
        for x, _y in trainloader:
            imgs.append(x)
            n += len(x)
            if n >= n_cal:
                break
        cal = torch.cat(imgs)[:n_cal]
        model = copy.deepcopy(base_model).eval()
        self.system = FaultTolerantNeuromorphic(model, cfg, calibration_images=cal)
        self.model = self.system.model.eval()
        self.bn_of = dict(bn_of)
        self.handles = {layer_of_handle(k): h for k, h in self.system.handles.items()}
        self.config = cfg
        sd = self.model.state_dict()
        self._mean0 = {
            bn: sd[f"{bn}.running_mean"].clone() for bn in set(self.bn_of.values())
            if f"{bn}.running_mean" in sd
        }  # fmt: skip
        self._mat0 = {k: h.accessor.read().clone() for k, h in self.handles.items()}

    def set_state(
        self,
        mats: Mapping[str, torch.Tensor] | None,
        bias: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Layers in `mats` are written; every other layer, and every BN mean not hit by
        `bias`, is reset to the digital checkpoint state (so calls are independent)."""
        mats = mats or {}
        bias = bias or {}
        for name, h in self.handles.items():
            m = mats.get(name)
            h.accessor.write(
                (self._mat0[name] if m is None else m.reshape(self._mat0[name].shape)).float()
            )
        sd = self.model.state_dict()
        for bn, m0 in self._mean0.items():
            sd[f"{bn}.running_mean"].copy_(m0)
        for name, b in bias.items():
            bn = self.bn_of[name.removesuffix(".weight")]
            sd[f"{bn}.running_mean"].sub_(b.to(sd[f"{bn}.running_mean"].dtype))

    @torch.no_grad()
    def acc(self, xe: torch.Tensor, ye: torch.Tensor, bs: int = 250) -> float:
        self.model.eval()
        correct = 0
        for i in range(0, len(xe), bs):
            correct += int((self.model(xe[i : i + bs]).argmax(1) == ye[i : i + bs]).sum())
        return 100.0 * correct / len(xe)


def aihwkit_native(
    model: torch.nn.Module,
    digital_head: str | None = "fc",
    g_max: float = 25.0,
    read_noise_scale: float = 1.0,
    seed: int = 0,
):
    """AIHWKit-converted copy of `model`: TorchInferenceRPUConfig, PCMLikeNoiseModel(g_max),
    GlobalDriftCompensation, ideal forward (no ADC/DAC/forward noise, so only the weight-space
    drift pipeline differs from the digital twin). `digital_head` (a top-level attribute name) is
    kept digital because the digital twin leaves fc untouched. Returns the analog model after
    `program_analog_weights()` (seeded through torch's global RNG, which AIHWKit uses)."""
    from aihwkit.inference.compensation.drift import GlobalDriftCompensation
    from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.simulator.configs import TorchInferenceRPUConfig

    rc = TorchInferenceRPUConfig()
    rc.noise_model = PCMLikeNoiseModel(g_max=g_max, read_noise_scale=read_noise_scale)
    rc.drift_compensation = GlobalDriftCompensation()
    rc.forward.is_perfect = True
    digital = copy.deepcopy(getattr(model, digital_head)) if digital_head else None
    am = convert_to_analog(copy.deepcopy(model), rc)
    if digital is not None:
        setattr(am, digital_head, digital)
    am.eval()
    torch.manual_seed(seed)
    am.program_analog_weights()
    return am

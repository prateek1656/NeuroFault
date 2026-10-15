# ruff: noqa: C408
"""Worker-side code for a_exp2_adabs.py (Exp 2): network bridge with AdaBS done right, the
BN-free workload W2, and the per-(workload, seed, scenario) job.

AdaBS calibration data (finding, 2026-10-09): the checkpoints were trained with RandomCrop(pad 4)
+ flip + Cutout, so their BN running statistics are those of AUGMENTED data. Re-estimating BN on
CLEAN images moves the statistics off what the weights were trained for and costs 0.2-1.0 pt even
on the UNDRIFTED digital network (seed 101: -0.7; recal on crop-augmented train images instead is
+0.4..+0.7). a_common.NetBridge calibrates on the first 256 clean TEST images, which is what made
AdaBS look harmful in F0.3. Here AdaBS uses TRAIN images with the training augmentation (no test
leakage), nested subsets (first N of one fixed 2600-image pool).

W2 (BN-free): BN folded into the conv weights (scale rows by gamma/sqrt(var+eps)) and a digital
bias c before deployment; the analog tiles hold the folded weights. The BN modules are set to the
identity-plus-bias form (gamma 1, var 1-eps, mean 0, beta c), so there are no statistics left to
re-estimate and AdaBS is not applicable; the mean-bias correction of the compensators becomes an
edit of the digital bias (running_mean shift in W1, same code path).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_common as ac
import torch
from a_common import BN_OF, CONV_LAYERS, INCUMBENTS, TileCtx, TileRun, best_of, make_drift
from f0_common import REPO_ROOT

from neurofault.retraining.bn_recal import recalibrate_bn
from neurofault.spectral.baselines import bind_all
from neurofault.spectral.exp2 import adabs_images_per_tile, fold_conv_bn, split_batches
from neurofault.spectral.methods_a import FieldFit
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import extract_tiles, load_checkpoint
from neurofault.spectral.types import DTYPE, ReadNoise, Tile, TileSpec, make_generator

# patch positions per image at each conv layer (32x32 input; maxpool after layer1/2/3 convs)
POSITIONS = {
    "prep.0.weight": 1024,
    "layer1.0.0.weight": 1024,
    "res1.conv_a.weight": 256,
    "res1.conv_b.weight": 256,
    "layer2.0.0.weight": 256,
    "layer3.0.0.weight": 64,
    "res2.conv_a.weight": 16,
    "res2.conv_b.weight": 16,
}
ADA_N_DEFAULT = 256


class Exp2Bridge:
    """ResNet-9 weight-space evaluation; W1 (BN kept) or W2 (BN folded into weights + bias)."""

    def __init__(self, ckpt_seed: int, workload: str, n_eval: int, cal_n: int = 2600):
        import torchvision

        from neurofault.models.cnn import SimpleCNN
        from neurofault.spectral.tiles import checkpoint_path

        T = torchvision.transforms
        self.workload = workload
        self.model = SimpleCNN()
        blob = torch.load(checkpoint_path(ckpt_seed), map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["model_state_dict"])
        self.model.eval()
        self.folded: dict[str, torch.Tensor] = {}
        if workload == "W2":
            self._fold()
        self.orig = {k: v.clone() for k, v in self.model.state_dict().items()}
        norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
        test = torchvision.datasets.CIFAR10(
            REPO_ROOT / "data", train=False, download=False,
            transform=T.Compose([T.ToTensor(), norm]),
        )  # fmt: skip
        self.xe = torch.stack([test[i][0] for i in range(n_eval)])
        self.ye = torch.tensor([test[i][1] for i in range(n_eval)])
        self.xc = None
        if workload == "W1":
            aug = T.Compose([
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm,
                T.RandomErasing(p=1.0, scale=(0.0156, 0.0156), ratio=(1.0, 1.0), value=0),
            ])  # fmt: skip
            train = torchvision.datasets.CIFAR10(
                REPO_ROOT / "data", train=True, download=False, transform=aug
            )
            g = torch.Generator().manual_seed(7000 + ckpt_seed)
            idx = torch.randperm(len(train), generator=g)[:cal_n].tolist()
            torch.manual_seed(7000 + ckpt_seed)  # augmentation draws
            self.xc = torch.stack([train[i][0] for i in idx])

    def _fold(self) -> None:
        sd = self.model.state_dict()
        for conv, bn in BN_OF.items():
            w, c = fold_conv_bn(
                sd[f"{conv}.weight"], sd[f"{bn}.weight"], sd[f"{bn}.bias"],
                sd[f"{bn}.running_mean"], sd[f"{bn}.running_var"], 1e-5,
            )  # fmt: skip
            self.folded[f"{conv}.weight"] = w.detach().flatten(1).to(DTYPE)
            sd[f"{conv}.weight"].copy_(w)
            sd[f"{bn}.weight"].fill_(1.0)
            sd[f"{bn}.bias"].copy_(c)
            sd[f"{bn}.running_mean"].zero_()
            sd[f"{bn}.running_var"].fill_(1.0 - 1e-5)  # sqrt(var + eps) == 1
        self.model.load_state_dict(sd)

    def layer_mats(self) -> dict[str, torch.Tensor]:
        """Deployed (flattened) matrices: folded for W2, checkpoint for W1."""
        return self.folded or load_checkpoint_cached(self._seed_key)

    @torch.no_grad()
    def acc(self, mats, bias=None, ada_n: int = 0) -> float:
        self.model.load_state_dict(self.orig)
        sd = self.model.state_dict()
        for name, w in (mats or {}).items():
            sd[name].copy_(w.reshape(sd[name].shape).to(sd[name].dtype))
            if bias and name in bias:
                sd[f"{BN_OF[name.removesuffix('.weight')]}.running_mean"].sub_(
                    bias[name].to(sd[name].dtype)
                )
        if ada_n:
            assert self.xc is not None, "AdaBS is not applicable to the BN-free workload"
            xs, s = [], 0
            for b in split_batches(ada_n):
                xs.append((self.xc[s : s + b], None))
                s += b
            recalibrate_bn(self.model, xs, len(xs), "cpu")
        self.model.eval()
        correct = 0
        for i in range(0, len(self.xe), 500):
            correct += int(
                (self.model(self.xe[i : i + 500]).argmax(1) == self.ye[i : i + 500]).sum()
            )
        return 100.0 * correct / len(self.xe)


_CK: dict[int, dict] = {}


def load_checkpoint_cached(seed: int) -> dict:
    if seed not in _CK:
        _CK[seed] = load_checkpoint(seed)
    return _CK[seed]


class Run2(TileRun):
    """TileRun with the incumbents plus one FieldFit('cos', order 4) per probe budget B
    (B = forward + backward probe MVMs per update, split evenly)."""

    def __init__(self, ctx: TileCtx, real, sigma: float, seed: int, budgets: list[int]):
        self.ctx, self.real, self.sigma = ctx, real, sigma
        self.cur = [ctx.stp]
        noise = ReadNoise("per_cell", sigma, adc_bits=None, dac_bits=None)
        gen = make_generator(seed, f"read:{ctx.key}")
        self.reader = TileReader(lambda: self.cur[0], noise, gen)
        sd = seed + (sum(ord(ch) for ch in repr(ctx.key)) & 0xFFFF)
        from neurofault.spectral.baselines import GDC, BiScale, PerColumn, PerRow

        self.methods = {
            "gdc": GDC("ls", p=8, seed=sd),
            "row": PerRow("rademacher", 8, "ls", seed=sd),
            "col": PerColumn("rademacher", 8, "backward", seed=sd),
            "bi": BiScale(8, 8, seed=sd),
        }
        for b in budgets:
            self.methods[f"m4_{b}"] = FieldFit("cos", order=4, p_fwd=b // 2, p_bwd=b // 2, seed=sd)
        bind_all(list(self.methods.values()), lambda: self.cur[0].weights(), lambda: ctx.tile.w)
        self.cost = {k: m.calibrate(self.reader) for k, m in self.methods.items()}
        self.y0bar = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        self.sketch = None


_SPEC = re.compile(r"^(none|gdc|col|b8|ada(\d+)|col_ada(\d+)|b8_ada(\d+)|m4_(\d+)|m4ada_(\d+))$")


def parse_spec(name: str, ada_stack_n: int = ADA_N_DEFAULT):
    """name -> (base method, uses mean-bias, AdaBS images)."""
    m = _SPEC.match(name)
    if not m:
        raise ValueError(name)
    if name in ("none", "gdc", "col"):
        return name, False, 0
    if name == "b8":
        return "b8", True, 0
    if name.startswith("col_ada"):
        return "col", False, int(name[7:])
    if name.startswith("b8_ada"):
        return "b8", False, int(name[6:])
    if name.startswith("ada"):
        return "gdc", False, int(name[3:])
    if name.startswith("m4ada_"):
        return f"m4_{name[6:]}", False, ada_stack_n
    return name, True, 0


def _tiles(seed: int):
    return extract_tiles(seed, CONV_LAYERS, 256, min_dim=16, tile_selection="all")


def mvms_per_image(seed: int) -> int:
    tiles = _tiles(seed)
    pos, cnt = [], []
    for layer in CONV_LAYERS:
        pos.append(POSITIONS[layer])
        cnt.append(sum(1 for t in tiles if t.spec.layer == layer))
    return adabs_images_per_tile(pos, cnt)


def _mk_get_tile(bridge: Exp2Bridge):
    def get_tile(ckpt_seed, layer, r0, c0, m, n, size):
        w = bridge.layer_mats()[layer][r0 : r0 + m, c0 : c0 + n].clone()
        return Tile(TileSpec(ckpt_seed, layer, r0, c0, m, n, size), w)

    return get_tile


def _setup(point: dict):
    cfg, seed, wl = point["cfg"], point["seed"], point["workload"]
    torch.manual_seed(0)
    bridge = Exp2Bridge(seed, wl, cfg["n_eval"])
    bridge._seed_key = seed
    old = ac.get_tile
    ac.get_tile = _mk_get_tile(bridge)
    return cfg, seed, wl, bridge, old


def oracle_job(point: dict) -> list[dict]:
    """Digital float accuracy and as-programmed (no drift) accuracy: B7 oracle = known E."""
    cfg, seed, wl, bridge, old = _setup(point)
    try:
        mats = bridge.layer_mats()
        pm = {}
        for tl in _tiles(seed):
            sp = tl.spec
            ctx = TileCtx(seed, sp.layer, sp.r0, sp.c0, sp.m, sp.n, sp.tile_size)
            pm.setdefault(sp.layer, torch.zeros_like(mats[sp.layer], dtype=torch.float32))
            pm[sp.layer][sp.r0 : sp.r0 + sp.m, sp.c0 : sp.c0 + sp.n] = ctx.wp.float()
        out = [dict(config="ref_digital", acc=bridge.acc(None), cost_mvm=0),
               dict(config="oracle", acc=bridge.acc(pm), cost_mvm=0)]  # fmt: skip
        if wl == "W1":
            out.append(dict(config="oracle_ada256", acc=bridge.acc(pm, None, 256), cost_mvm=0))
            out.append(
                dict(config="ref_digital_ada256", acc=bridge.acc(None, None, 256), cost_mvm=0)
            )
        return [dict(out_row, n_eval=cfg["n_eval"]) for out_row in out]
    finally:
        ac.get_tile = old


def net_job(point: dict) -> list[dict]:
    """One (workload, ckpt seed, drift scenario): accuracy + counted cost for every requested
    config at every time."""
    cfg, seed, wl, bridge, old = _setup(point)
    sc = point["scenario"]
    try:
        names = [n for n in cfg["evalsets"][wl][sc["tier"]] if wl == "W1" or "ada" not in n]
        budgets = sorted({int(n.split("_")[1]) for n in names if n.startswith(("m4_", "m4ada_"))})
        specs = {n: parse_spec(n) for n in names}
        mats_all = bridge.layer_mats()
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs = []
        for tl in _tiles(seed):
            sp = tl.spec
            ctx = TileCtx(seed, sp.layer, sp.r0, sp.c0, sp.m, sp.n, sp.tile_size)
            real = mod.sample(ctx.stp, make_generator(seed, f"drift:{sp.layer}:{sp.r0}:{sp.c0}"))
            runs.append((sp, ctx.slim(), Run2(ctx, real, cfg["sigma"], seed, budgets)))
        per_img = mvms_per_image(seed)
        rows = []
        bases = {s[0] for s in specs.values()}
        for tlabel, t in cfg["times"].items():
            W = {b: {} for b in bases}
            B = {b: {} for b in bases}
            cost = dict.fromkeys(bases, 0)
            bias_cost = dict.fromkeys(bases, 0)
            for sp, _ctx, run in runs:
                res = run.at(t)
                for base in bases:
                    key = best_of(res, INCUMBENTS) if base == "b8" else base
                    r = res[key]
                    w = r["W"] if key != "none" else res["_wt"]
                    W[base].setdefault(
                        sp.layer, torch.zeros_like(mats_all[sp.layer], dtype=torch.float32)
                    )
                    W[base][sp.layer][sp.r0 : sp.r0 + sp.m, sp.c0 : sp.c0 + sp.n] = w.float()
                    uses_bias = any(s[0] == base and s[1] for s in specs.values())
                    if uses_bias and key != "none":
                        bd = B[base].setdefault(sp.layer, torch.zeros(mats_all[sp.layer].shape[0]))
                        bd[sp.r0 : sp.r0 + sp.m] += r["b"].float()
                    c = r.get("cost")
                    n_mvm = 0 if c is None else int(c.mvm_fwd + c.mvm_bwd)
                    cost[base] += n_mvm
                    bias_cost[base] += 1 if uses_bias and key != "none" else 0
            for name, (base, use_b, ada_n) in specs.items():
                acc = bridge.acc(W[base], B[base] if use_b else None, ada_n)
                rows.append(dict(
                    workload=wl, time=tlabel, t_s=t, config=name, acc=acc,
                    cost_mvm=cost[base] + (bias_cost[base] if use_b else 0) + ada_n * per_img,
                    probe_mvm=cost[base],
                    ada_images=ada_n, n_eval=cfg["n_eval"],
                ))  # fmt: skip
            del W, B
        return rows
    finally:
        ac.get_tile = old

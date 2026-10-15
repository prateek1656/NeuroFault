# ruff: noqa: C408
"""Shared engine for the Direction-A scripts F0.2, F0.3 and Gate 0'.

One tile engine (TileCtx + run_tile) and one network bridge (NetBridge) so the three scripts
measure the same things the same way. Nothing here judges a number; criteria files do.

Conventions / deliberate choices (all reported in the run config):
- Encoding diff_pair g_min_frac 0.01, programming noise aihwkit_pcm_poly. ALL errors are measured
  against Wp = the as-programmed weights at t0 (drift error only; programming error is a floor
  common to every method). Estimators never see Wp or W(t): they read through a counted TileReader;
  W(t) is used only to materialise W_eff (baselines.base docstring); M4 additionally uses the
  stored digital reference W0 = checkpoint tile (the programming target).
- Output error: ||(W_eff - Wp) X + b 1^T|| / ||Wp X|| on held-out cached im2col columns X_eval
  (columns 2048: of results/spectral/activations/<seed>/<layer>.pt X_cache); the weighting
  covariances / PCs are fitted on columns :2048.
- "B8" = per-tile best of {GDC, per-row, per-column, bi-scale} each + mean-bias correction, chosen
  by smallest held-out error (selection oracle: optimistic for the incumbent, so headroom is
  conservative). "M4best" = per-tile best of {cos, rank-1} + bias, chosen the same way.
- Mean-bias correction: b = y0bar - (read(xbar) + (W_eff - W_t) xbar) with xbar the layer mean
  activation (2 MVMs: one stored at t0, one now).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from f0_common import REPO_ROOT, get_tile

from neurofault.spectral import metrics as M
from neurofault.spectral.baselines import (
    GDC,
    BiScale,
    GDCJoshi,
    PerColumn,
    PerRow,
    ProbeLowRank,
    TimerSchedule,
    bind_all,
)
from neurofault.spectral.drift import (
    PCMPerDevice,
    PeripheryGain,
    PowerLawUniform,
    RRAMRelaxation,
    StructuredNu,
    ThermalGain,
    pcm_nu_stats,
)
from neurofault.spectral.encoding import encode
from neurofault.spectral.methods_a import FieldFit, InputSubspace, ResidualSketch
from neurofault.spectral.noise import program
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import extract_tiles
from neurofault.spectral.types import (
    DTYPE,
    T0_S,
    EncodingSpec,
    ProgNoise,
    ReadNoise,
    make_generator,
)

CONV_LAYERS = [
    "prep.0.weight",
    "layer1.0.0.weight",
    "res1.conv_a.weight",
    "res1.conv_b.weight",
    "layer2.0.0.weight",
    "layer3.0.0.weight",
    "res2.conv_a.weight",
    "res2.conv_b.weight",
]
BN_OF = {
    "prep.0": "prep.1",
    "layer1.0.0": "layer1.0.1",
    "res1.conv_a": "res1.bn_a",
    "res1.conv_b": "res1.bn_b",
    "layer2.0.0": "layer2.0.1",
    "layer3.0.0": "layer3.0.1",
    "res2.conv_a": "res2.bn_a",
    "res2.conv_b": "res2.bn_b",
}
TIMES = {"1h": 3600.0, "1d": 86400.0, "30d": 30 * 86400.0, "1yr": 365.25 * 86400.0}
ENC = EncodingSpec("diff_pair", g_min_frac=0.01)
PROG = ProgNoise("aihwkit_pcm_poly")
N_FIT = 2048
INCUMBENTS = ("gdc", "row", "col", "bi")
D3_FIELDS = ("row_gradient", "col_gradient", "row_plus_col", "rank_r", "hotspot")

_ACT: dict[tuple[int, str], dict] = {}


def act_stats(seed: int, layer: str) -> dict:
    key = (seed, layer)
    if key not in _ACT:
        path = (
            REPO_ROOT / "results" / "spectral" / "activations" / str(seed)
            / f"{layer.removesuffix('.weight')}.pt"
        )  # fmt: skip
        _ACT[key] = torch.load(path, weights_only=False)
    return _ACT[key]


# --------------------------------------------------------------------------- drift factory


def make_drift(model: str, field: str = "row_gradient", rho: float = 0.0, scale: float = 1.0):
    """model: D1 | D2 | D3 | D4 | D4rho | D5a | D5b | D7out | D7in."""
    if model == "D1":
        return PowerLawUniform()
    if model == "D2":
        return PCMPerDevice(drift_scale=scale)
    if model == "D3":
        return StructuredNu(field=field, rho=rho, drift_scale=scale)
    if model == "D4":  # assumed defaults: 330 K, Ea 0.5 eV, mixed dT field, over D2
        return ThermalGain(base=PCMPerDevice(drift_scale=scale))
    if model == "D4rho":  # temperature-gradient amplitude axis: dT_amp = 20 K * rho, no Arrhenius
        return ThermalGain(
            base=PCMPerDevice(drift_scale=scale), T_K=300.0, dT_amp_K=20.0 * rho, dT_kind="mixed"
        )
    if model == "D5a":
        return RRAMRelaxation("cmo")
    if model == "D5b":
        return RRAMRelaxation("mean_revert")
    if model == "D7out":
        return PeripheryGain(axis="out")
    if model == "D7in":
        return PeripheryGain(axis="in")
    raise ValueError(model)


# --------------------------------------------------------------------------- tile context


def _psd_sqrt(c: torch.Tensor) -> torch.Tensor:
    ev, v = torch.linalg.eigh((c + c.T) / 2)
    return (v * ev.clamp_min(0).sqrt()) @ v.T


class TileCtx:
    """Programmed tile + everything that weights or evaluates a correction on it."""

    def __init__(self, ckpt_seed: int, layer: str, r0: int, c0: int, m: int, n: int, size: int):
        self.key = (ckpt_seed, layer, r0, c0)
        self.tile = get_tile(ckpt_seed, layer, r0, c0, m, n, size)
        self.st0 = encode(self.tile, ENC)
        gp = make_generator(ckpt_seed, f"prog:{layer}:{r0}:{c0}")
        self.stp = program(self.st0, PROG, gp)
        self.wp = self.stp.weights()
        st = act_stats(ckpt_seed, layer)
        x = st["X_cache"][c0 : c0 + n].to(DTYPE)
        self.x_fit, self.x_eval = x[:, :N_FIT], x[:, N_FIT:]
        self.xbar = self.x_fit.mean(1)
        c_unc = self.x_fit @ self.x_fit.T / self.x_fit.shape[1]
        xc = self.x_fit - self.xbar[:, None]
        c_cen = xc @ xc.T / xc.shape[1]
        self.cs, self.cs_c = _psd_sqrt(c_unc), _psd_sqrt(c_cen)
        self.pcs_unc = torch.linalg.eigh(c_unc)[1].flip(1)
        self.pcs_cen = torch.linalg.eigh(c_cen)[1].flip(1)
        self.var_frac8 = float(torch.linalg.eigvalsh(c_cen).flip(0)[:8].sum() / torch.trace(c_cen))
        self.var_frac8_unc = float(
            torch.linalg.eigvalsh(c_unc).flip(0)[:8].sum() / torch.trace(c_unc)
        )
        self.ref_out = self.wp @ self.x_eval
        self.ref_norm = float(self.ref_out.norm())

    def slim(self) -> TileCtx:
        """Drop what only construction needs (keeps a 109-tile network job under ~0.5 GB)."""
        for a in ("x_fit", "pcs_unc", "pcs_cen", "cs", "ref_out"):
            setattr(self, a, None)
        return self

    def err(self, w_eff: torch.Tensor, b: torch.Tensor | None = None) -> float:
        d = (w_eff - self.wp) @ self.x_eval
        if b is not None:
            d = d + b[:, None]
        return float(d.norm() / self.ref_norm)


def select_tiles(seeds, layers, cap: int, tile_size: int = 256) -> list[tuple]:
    out = []
    for s in seeds:
        for t in extract_tiles(
            s, layers, tile_size, min_dim=16, max_tiles_per_layer=cap, tile_selection="random"
        ):
            sp = t.spec
            out.append((s, sp.layer, sp.r0, sp.c0, sp.m, sp.n, sp.tile_size))
    return out


# --------------------------------------------------------------------------- method engine


def gains_of(meta: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(row multiplier g incl. alpha, column multiplier h) of a gain-type correction."""
    g = meta.get("row_gain")
    h = meta.get("col_gain")
    if "alpha" in meta:
        g = meta["alpha"] if g is None else g * meta["alpha"]
        if not torch.is_tensor(g):
            g = torch.full((1,), float(g), dtype=DTYPE)
    return g, h


class TileRun:
    """Calibrated method set on one tile + drift realization; update(t) yields corrections."""

    def __init__(
        self,
        ctx: TileCtx,
        real,
        sigma: float,
        seed: int,
        methods: tuple[str, ...] = ("gdc", "row", "col", "bi", "m4cos", "m4r1"),
        p: int = 8,
        extra_probe_methods: bool = False,
    ):
        self.ctx, self.real, self.sigma = ctx, real, sigma
        self.cur = [ctx.stp]
        noise = ReadNoise("per_cell", sigma, adc_bits=None, dac_bits=None)
        gen = make_generator(seed, f"read:{ctx.key}")
        self.reader = TileReader(lambda: self.cur[0], noise, gen)
        tag = sum(ord(ch) for ch in repr(ctx.key)) & 0xFFFF
        sd = seed + tag
        mk = {
            "gdc": lambda: GDC("ls", p=p, seed=sd),
            "row": lambda: PerRow("rademacher", p, "ls", seed=sd),
            "col": lambda: PerColumn("rademacher", p, "backward", seed=sd),
            "bi": lambda: BiScale(p, p, seed=sd),
            "m4cos": lambda: FieldFit("cos", order=4, p_fwd=p, p_bwd=p, seed=sd),
            "m4r1": lambda: FieldFit("rank", rank=1, p_fwd=p, p_bwd=p, iters=10, seed=sd),
            "m3": lambda: InputSubspace(ctx.pcs_unc[:, :p]),
            "b5": lambda: ProbeLowRank(k=p // 2, s=p - p // 2, seed=sd),
            "b5s": lambda: ProbeLowRank(k=2, s=2, seed=sd),  # k <= 4, P <= 4 (H3.5)
        }
        self.methods = {k: mk[k]() for k in methods}
        bind_all(list(self.methods.values()), lambda: self.cur[0].weights(), lambda: ctx.tile.w)
        self.cost = {k: m.calibrate(self.reader) for k, m in self.methods.items()}
        n_ = ctx.xbar.shape[0]
        self.y0bar = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        self.sketch = ResidualSketch(k=8, s=4, seed=sd) if extra_probe_methods else None
        if self.sketch is not None:
            self.sketch.calibrate(self.reader)
        _ = n_

    def at(self, t: float) -> dict:
        """{method: dict(W, b, meta, cost)} at time t, plus '_wt', '_xread'."""
        ctx = self.ctx
        self.cur[0] = self.real.at(t)
        wt = self.cur[0].weights()
        xread = self.reader.mvm(ctx.xbar[:, None])[:, 0]
        out = {"_wt": wt}
        for k, m in self.methods.items():
            c = m.update(self.reader, t)
            b = self.y0bar - (xread + (c.W_eff - wt) @ ctx.xbar)
            out[k] = dict(W=c.W_eff, b=b, meta=c.meta, cost=c.cost, err=ctx.err(c.W_eff, b))
            out[k]["err_nb"] = ctx.err(c.W_eff)
        out["none"] = dict(W=wt, b=None, meta={}, err=ctx.err(wt), err_nb=ctx.err(wt))
        return out


def best_of(res: dict, names) -> str:
    return min((n for n in names if n in res), key=lambda n: res[n]["err"])


def weighted_capture(r: torch.Tensor, cs: torch.Tensor, k: int, basis=None) -> float:
    return M.rank_energy(r, [k], C_sqrt=cs, basis=basis)[k]


def rank_k_project(r: torch.Tensor, cs: torch.Tensor, k: int) -> torch.Tensor:
    """Best rank-k output-side projection of r under the weighting cs: U_k U_k^T r."""
    u = torch.linalg.svd(r @ cs, full_matrices=False)[0][:, :k]
    return u @ (u.T @ r)


def aucs(pos_neg_scores: dict[str, np.ndarray], label: np.ndarray) -> dict[str, float]:
    out = {}
    for k, s in pos_neg_scores.items():
        if label.all() or (~label).all():
            out[k] = float("nan")
        else:
            out[k] = M.auc(s[label], s[~label])
    return out


def partial_auc_given(score: np.ndarray, cond: np.ndarray, label: np.ndarray, bins: int = 5):
    """AUC of `score` within quantile bins of `cond` (the free GDC factor), weighted by the
    number of (pos, neg) pairs per bin: the part of the signal NOT explained by `cond`."""
    edges = np.quantile(cond, np.linspace(0, 1, bins + 1))
    num = den = 0.0
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else np.inf
        sel = (cond >= edges[i]) & (cond < hi)
        lab = label[sel]
        npos, nneg = int(lab.sum()), int((~lab).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * M.auc(score[sel][lab], score[sel][~lab])
        den += w
    return num / den if den else float("nan")


# --------------------------------------------------------------------------- network bridge

_NET: dict[tuple[int, str], NetBridge] = {}


class NetBridge:
    """Weight-space ResNet-9 evaluation: conv matrices (and mean-bias via BN running_mean) are
    replaced, fc and BN affine stay digital. AdaBS = recalibrate_bn on the calibration images.

    cal="test_clean" (legacy, kept for provenance of the first F0.3 run): first n_cal CLEAN test
    images. This is biased: the checkpoints were trained with crop/flip/Cutout, so their BN stats
    are those of augmented data and clean-image re-estimation costs ~0.2-1 pt even undrifted.
    cal="train_aug" (fixed, ; same pool/seeds as a_exp2_jobs.Exp2Bridge): n_cal images
    from a fixed 2600-image pool of AUGMENTED TRAIN images (no test leakage). The eval set
    (test images n_cal..n_cal+n_eval) is the same in both modes."""

    def __init__(self, ckpt_seed: int, n_eval: int, n_cal: int = 256, cal: str = "test_clean"):
        import torchvision

        from neurofault.models.cnn import SimpleCNN
        from neurofault.spectral.tiles import checkpoint_path

        self.model = SimpleCNN()
        blob = torch.load(checkpoint_path(ckpt_seed), map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["model_state_dict"])
        self.model.eval()
        self.orig = {k: v.clone() for k, v in self.model.state_dict().items()}
        tf = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )  # fmt: skip
        ds = torchvision.datasets.CIFAR10(REPO_ROOT / "data", train=False, download=False,
                                          transform=tf)  # fmt: skip
        xs = torch.stack([ds[i][0] for i in range(n_cal + n_eval)])
        ys = torch.tensor([ds[i][1] for i in range(n_cal + n_eval)])
        self.xc, self.xe, self.ye = xs[:n_cal], xs[n_cal:], ys[n_cal:]
        if cal == "train_aug":
            T = torchvision.transforms
            norm = T.Normalize((0.5,) * 3, (0.5,) * 3)
            aug = T.Compose([
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm,
                T.RandomErasing(p=1.0, scale=(0.0156, 0.0156), ratio=(1.0, 1.0), value=0),
            ])  # fmt: skip
            train = torchvision.datasets.CIFAR10(REPO_ROOT / "data", train=True, download=False,
                                                 transform=aug)  # fmt: skip
            g = torch.Generator().manual_seed(7000 + ckpt_seed)
            idx = torch.randperm(len(train), generator=g)[:2600].tolist()
            torch.manual_seed(7000 + ckpt_seed)  # augmentation draws (as Exp2Bridge)
            self.xc = torch.stack([train[i][0] for i in idx])[:n_cal]
        elif cal != "test_clean":
            raise ValueError(f"unknown AdaBS calibration source {cal!r}")

    @torch.no_grad()
    def acc(self, mats: dict[str, torch.Tensor] | None, bias=None, adabs: bool = False) -> float:
        from neurofault.retraining.bn_recal import recalibrate_bn

        self.model.load_state_dict(self.orig)
        sd = self.model.state_dict()
        for name, w in (mats or {}).items():
            sd[name].copy_(w.reshape(sd[name].shape).to(sd[name].dtype))
            if bias and name in bias:
                bn = BN_OF[name.removesuffix(".weight")]
                sd[f"{bn}.running_mean"].sub_(bias[name].to(sd[name].dtype))
        if adabs:
            loader = [(self.xc, torch.zeros(len(self.xc)))]
            recalibrate_bn(self.model, loader, 1, "cpu")
        self.model.eval()
        correct = 0
        for i in range(0, len(self.xe), 500):
            correct += int(
                (self.model(self.xe[i : i + 500]).argmax(1) == self.ye[i : i + 500]).sum()
            )
        return 100.0 * correct / len(self.xe)


def net_bridge(ckpt_seed: int, n_eval: int, cal: str = "test_clean") -> NetBridge:
    if (ckpt_seed, cal) not in _NET:
        _NET[(ckpt_seed, cal)] = NetBridge(ckpt_seed, n_eval, cal=cal)
    return _NET[(ckpt_seed, cal)]


def joshi_ref_current(stp, scale: float, gen: torch.Generator, n_ref: int = 8, frac: float = 0.5):
    """Sum current of n_ref reference PCM devices at frac*g_max drifting under the SAME D2 model
    (nu drawn once per device, drift_scale = the true scale)."""
    g = torch.full((n_ref,), frac * stp.enc.g_max_uS, dtype=DTYPE)
    mu, sig = pcm_nu_stats(g, stp.enc.g_max_uS)
    nu = (mu + sig * torch.randn(n_ref, generator=gen, dtype=DTYPE)).abs() * scale
    return lambda t: float((g * (((t + T0_S) / T0_S) ** (-nu))).sum())


# --------------------------------------------------------------------------- tile-level conditions


def scenario_list(cfg: dict) -> list[dict]:
    sc = [dict(model="D2", field="", rho=0.0, scale=1.0)]
    for f in cfg.get("fields", []):
        sc += [dict(model="D3", field=f, rho=float(r), scale=1.0) for r in cfg["rhos"]]
    if cfg.get("d4"):
        sc += [dict(model="D4rho", field="", rho=float(r), scale=1.0) for r in cfg["rhos"]]
    for s_ in cfg.get("d2_scales", []):
        sc.append(dict(model="D2", field="", rho=0.0, scale=float(s_)))
    return sc


def jacobian_rank_deficit(ctx: TileCtx, seed: int, t: float = 86400.0, beta: float = 1e-3) -> dict:
    """Numerical Jacobian of the M1 signature s = [sigma1/sigma1(0), sigma1/sigma2, 1-|<v1,v1(0)>|]
    w.r.t. latents (L = ln((t+t0)/t0), aging = nu scale, dT uniform temperature offset) on a D2
    realization; rank from singular values of the column-normalised J (tol 1e-6)."""
    real = make_drift("D2").sample(ctx.stp, make_generator(seed, "jac"))
    gp, gn, nu = ctx.stp.g_pos, ctx.stp.g_neg, real.nu
    _u0, s0, vh0 = torch.linalg.svd(ctx.wp)

    def sig(th):
        f = torch.exp(-nu * th[1] * th[0]) * (1.0 + beta * th[2])
        w = (gp * f[..., 0] - gn * f[..., 1]) / ctx.stp.scale
        _u, s, vh = torch.linalg.svd(w)
        return torch.stack([s[0] / s0[0], s[0] / s[1], 1.0 - (vh[0] @ vh0[0]).abs()])

    th0 = torch.tensor([float(np.log((t + T0_S) / T0_S)), 1.0, 0.0], dtype=DTYPE)
    cols = []
    for i in range(3):
        h = 1e-4 * max(abs(float(th0[i])), 1.0)
        d = torch.zeros(3, dtype=DTYPE)
        d[i] = h
        cols.append((sig(th0 + d) - sig(th0 - d)) / (2 * h))
    j = torch.stack(cols, 1)
    j = j / j.norm(dim=0, keepdim=True).clamp(min=1e-300)
    sv = torch.linalg.svdvals(j)
    rank = int((sv / sv.max() > 1e-6).sum())
    return dict(
        jacobian_rank=rank,
        jacobian_rank_deficit=3 - rank,
        jacobian_sv_min_ratio=float(sv[-1] / sv[0]),
    )


def tile_conditions(point: dict) -> list[dict]:
    """Raw per-(tile, scenario, noise, time) rows: errors of every method, B8 / M4 selections,
    centred captures (cfg.captures) and watchdog statistics (noise-free rows)."""
    cfg = point["cfg"]
    seed, layer, r0, c0, m, n, size = point["tile"]
    ctx = TileCtx(seed, layer, r0, c0, m, n, size)
    sv_p = torch.linalg.svdvals(ctx.wp)
    methods = tuple(cfg["methods"])
    rows = []
    if cfg.get("jacobian"):
        rows.append(dict(kind="jac", seed=seed, layer=layer, tile=f"{r0}_{c0}",
                         **jacobian_rank_deficit(ctx, seed)))  # fmt: skip
    for sc in scenario_list(cfg):
        mod = make_drift(sc["model"], sc["field"] or "row_gradient", sc["rho"], sc["scale"])
        real = mod.sample(ctx.stp, make_generator(seed, f"drift:{layer}:{r0}:{c0}"))
        for sigma in cfg["sigmas"]:
            run = TileRun(ctx, real, sigma, seed, methods, cfg.get("p", 8),
                          extra_probe_methods=bool(cfg.get("captures")))  # fmt: skip
            for tl in cfg["times"]:
                t = TIMES[tl]
                res = run.at(t)
                wt = res["_wt"]
                row = dict(kind="cond", seed=seed, layer=layer, tile=f"{r0}_{c0}", sigma=sigma,
                           time=tl, t_s=t, **sc)  # fmt: skip
                for k in ("none", *methods):
                    row[f"err_{k}"] = res[k]["err"]
                b8 = best_of(res, INCUMBENTS)
                row["b8_name"] = b8
                row["err_b8"] = res[b8]["err"]
                m4 = [k for k in ("m4cos", "m4r1") if k in res]
                if m4:
                    k4 = best_of(res, m4)
                    row["m4_name"], row["err_m4best"] = k4, res[k4]["err"]
                if sigma == 0.0:
                    sv = torch.linalg.svdvals(wt)
                    row["s1_shift"] = abs(float(sv[0] / sv_p[0]) - 1.0)
                    row["alpha_dev"] = abs(float(res["gdc"]["meta"]["alpha"]) - 1.0)
                    row["cur_sum_dev"] = abs(float((wt - ctx.wp).sum(1).sum()))
                if cfg.get("captures"):
                    r = res[b8]["W"] - ctx.wp
                    for k in cfg["cap_ks"]:
                        row[f"cap_m2_k{k}"] = weighted_capture(r, ctx.cs_c, k)
                        row[f"cap_m3_k{k}"] = weighted_capture(r, ctx.cs_c, k, ctx.pcs_cen)
                    g, h = gains_of(res[b8]["meta"])
                    r_hat, _ = run.sketch.update(run.reader, t, g, h)
                    num = float(((r - r_hat) @ ctx.cs_c).pow(2).sum())
                    row["cap_m2_real_k8"] = 1.0 - num / max(
                        float((r @ ctx.cs_c).pow(2).sum()), 1e-300
                    )
                rows.append(row)
    return rows


def watchdog_auc_rows(cond: pd.DataFrame) -> pd.DataFrame:  # noqa: F821
    """Per-tile AUCs over that tile's noise-free conditions. Label = B8 output error above the
    tile's median B8 error (stand-in for '1 pt equivalent', which needs the network)."""
    import pandas as pd

    out = []
    d0 = cond[cond.sigma == 0.0]
    for (seed, layer, tile), g in d0.groupby(["seed", "layer", "tile"]):
        lab = (g.err_b8 > g.err_b8.median()).to_numpy()
        if lab.all() or (~lab).all():
            continue
        s1, al, cs = (g[c].to_numpy() for c in ("s1_shift", "alpha_dev", "cur_sum_dev"))
        a = aucs({"s1": s1, "al": al, "cs": cs}, lab)
        out.append(dict(seed=seed, layer=layer, tile=tile, auc_sigma1_ratio=a["s1"],
                        auc_gdc_factor=a["al"], auc_current_sum=a["cs"],
                        auc_sigma1_minus_auc_gdc_factor=a["s1"] - a["al"],
                        m1_partial_auc_given_gdc=partial_auc_given(s1, al, lab)))  # fmt: skip
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- network job


def net_job(point: dict) -> list[dict]:
    """One (ckpt seed, drift scenario): accuracy of every method at every time, all tiles."""
    cfg = point["cfg"]
    seed, sc = point["seed"], point["scenario"]
    want = set(cfg["net_methods"])
    if sc["model"] != "D2":
        want -= {"timer", "b1a"}
    nb = net_bridge(seed, cfg["n_eval"], point.get("adabs_cal", "test_clean"))
    from neurofault.spectral.tiles import load_checkpoint

    full = load_checkpoint(seed)
    tiles = extract_tiles(seed, CONV_LAYERS, 256, min_dim=16, tile_selection="all")
    mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], sc["scale"])
    runs = []
    timer_nom = PCMPerDevice(drift_scale=1.0)
    for tl in tiles:
        sp = tl.spec
        ctx = TileCtx(seed, sp.layer, sp.r0, sp.c0, sp.m, sp.n, sp.tile_size)
        real = mod.sample(ctx.stp, make_generator(seed, f"drift:{sp.layer}:{sp.r0}:{sp.c0}"))
        run = TileRun(ctx, real, 0.0, seed, INCUMBENTS)
        extra = {}
        if {"timer", "b1a"} & want:
            tm = TimerSchedule.from_model(timer_nom, ctx.stp, ctx.wp, seed=seed)
            tm.bind(lambda r=run: r.cur[0].weights(), lambda c=ctx: c.tile.w)
            jb = GDCJoshi(8).bind(lambda r=run: r.cur[0].weights(), lambda c=ctx: c.tile.w)
            jb.bind_reference(
                joshi_ref_current(
                    ctx.stp, sc["scale"], make_generator(seed, f"jref:{sp.layer}:{sp.r0}:{sp.c0}")
                )
            )
            jb.calibrate(run.reader)
            extra = dict(timer=tm, b1a=jb)
        runs.append((sp, ctx.slim(), run, extra))
        _ = full
    rows = []
    base = dict(seed=seed, model=sc["model"], field=sc.get("field", ""), rho=sc["rho"],
                scale=sc["scale"], mismatch_factor=sc["scale"] if sc["model"] == "D2" else float("nan"))  # fmt: skip
    acc_ref = nb.acc(None)
    pm = {}
    for sp, ctx, _run, _ex in runs:
        pm.setdefault(sp.layer, torch.zeros_like(full[sp.layer], dtype=torch.float32))
        pm[sp.layer][sp.r0 : sp.r0 + sp.m, sp.c0 : sp.c0 + sp.n] = ctx.wp.float()
    acc_prog = nb.acc(pm)
    del pm
    for tlabel, t in cfg["net_times"].items():
        W = {k: {} for k in want | {"gdc"}}
        B = {k: {} for k in ("b8", "b8o16")}
        for sp, ctx, run, extra in runs:
            res = run.at(t)
            b8 = best_of(res, INCUMBENTS)
            outs = {"none": (res["_wt"], None), "gdc": (res["gdc"]["W"], None),
                    "row": (res["row"]["W"], None), "col": (res["col"]["W"], None),
                    "bi": (res["bi"]["W"], None), "b8": (res[b8]["W"], res[b8]["b"])}  # fmt: skip
            if "b8o16" in want:
                r = res[b8]["W"] - ctx.wp
                wo = res[b8]["W"] - rank_k_project(r, ctx.cs_c, 16)
                outs["b8o16"] = (wo, (ctx.wp - wo) @ ctx.xbar)
            for k, ex in extra.items():
                if k in want:
                    outs[k] = (ex.update(run.reader, t).W_eff, None)
            for k, (w, b) in outs.items():
                if k not in W and k not in want:
                    continue
                if sp.layer not in W.setdefault(k, {}):
                    W[k][sp.layer] = torch.zeros_like(full[sp.layer], dtype=torch.float32)
                W[k][sp.layer][sp.r0 : sp.r0 + sp.m, sp.c0 : sp.c0 + sp.n] = w.float()
                if b is not None:
                    bd = B.setdefault(k, {})
                    bd.setdefault(
                        sp.layer, torch.zeros(full[sp.layer].shape[0], dtype=torch.float32)
                    )
                    bd[sp.layer][sp.r0 : sp.r0 + sp.m] += b.float()
        row = dict(**base, time=tlabel, t_s=t, acc_ref=acc_ref, acc_prog=acc_prog)
        for k in want - {"gdc_adabs"}:
            row[f"acc_{k}"] = nb.acc(W[k], B.get(k), adabs=False)
        if "gdc_adabs" in want:
            row["acc_gdc_adabs"] = nb.acc(W["gdc"], None, adabs=True)
        rows.append(row)
        del W, B
    return rows


def coerce_numeric_strings(obj):
    """PyYAML reads '1.0e4' (no sign in the exponent) as a STRING. The frozen a_gate0.yaml has
    t_s thresholds written that way, which makes the verdict clause 'no data'. This returns a
    copy with numeric-looking strings turned into floats, for a clearly-labelled second verdict."""
    if isinstance(obj, dict):
        return {k: coerce_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj

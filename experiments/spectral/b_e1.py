"""Direction B, E1 (bidirectional vs forward-only read counts, C1/C1') and E1b (probe weight vs
noise model, C2). Arms P2.1, P2.2. Verdict thresholds live in the frozen
criteria/b_claims.yaml (never edited here); this script emits the metric names it references.

    python experiments/spectral/b_e1.py --exp e1|e1b [--smoke] [--allow-dirty] [--jobs 4]

Conventions follow b_e0_e1c.py: offset_refcol encoding, sigma_c as a fraction of max|w|,
Neyman-Pearson thresholds (fa), per-tile/ckpt-seeded noise; truth = faults with
|delta_w| >= visible_frac*max|w|. Every design is built in a forward-only form (reader without
backward support, so it cannot cheat) and a bidirectional form; C1 compares the best of each at
equal reads, "best" = fewest mean reads among configurations whose instance-mean F1 >= 0.9 (the
same best-of rule for both sides). Designs are told the true k (through their design parameters:
d per line, weights); that is the stated protocol of the plan, not an estimate of k.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.integrate import trapezoid
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.harness import run_grid, write_run
from neurofault.spectral.localize import (
    AdaptiveBisectionLocalizer,
    FullReadoutLocalizer,
    GroupTestingLocalizer,
    HPELocalizer,
    NonAdaptiveCSLocalizer,
    ResidualLocalizer,
    RothLocalizer,
    TwoStageLocalizer,
)
from neurofault.spectral.metrics import localization_f1
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import load_checkpoint
from neurofault.spectral.types import (
    DTYPE,
    EncodingSpec,
    FaultSpec,
    ReadNoise,
    Tile,
    TileSpec,
    make_generator,
)

CFG_DIR = Path(__file__).resolve().parent / "configs"


@lru_cache(maxsize=8)
def _ckpt(seed: int) -> dict:
    return load_checkpoint(seed)


# ------------------------------------------------------------------ design catalogue


PS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)  # total probes (A5)
TS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)  # total tests (A6)
A7_VARIANTS = ((0.25, 0.1), (0.5, 0.1), (0.5, 0.02), (0.5, 0.002))  # (delta_frac, SPRT err)


def A6_VARIANTS(n):
    """(tag, thresh, tol_frac, weight, delta_frac): strict-NP 1-bit tests at three weights."""
    return [("np2", "np", 0.0, n // 2, 0.25), ("np8", "np", 0.0, n // 8, 0.25),
            ("np32", "np", 0.0, n // 32, 0.25)]  # fmt: skip


def designs(w, k, fa, seed):
    """[(label, direction, config, builder)] for E1. direction in {fwd, bidir}."""
    n = w.shape[1]
    dc = 1 if k <= 4 else 2 if k <= 32 else 3  # designed max faults per line
    kw = {"fa": fa, "seed": seed}
    out = [("A1", "fwd", "full", lambda: FullReadoutLocalizer(w, **kw))]
    for r in (1, 2, 4, 8, 16, 32):
        out.append(
            (
                "A2",
                "fwd",
                f"r{r}",
                lambda r=r: HPELocalizer(w, n_dense=r, n_bisect=max(1, r // 2), **kw),
            )
        )
    for r in (2, 4, 8, 16):
        out.append(
            (
                "A2b",
                "fwd",
                f"r{r}",
                lambda r=r: RothLocalizer(
                    w, d_lines=1, d_cells=dc, n_dense=r, stage1="direct", **kw
                ),
            )
        )
    for nf, npr in ((2, 16), (4, 8), (4, 16), (4, 32), (4, 64), (8, 32)):
        out.append(
            (
                "A3",
                "fwd",
                f"f{nf}p{npr}",
                lambda nf=nf, npr=npr: ResidualLocalizer(w, n_fwd=nf, n_bwd=0, n_prune=npr, **kw),
            )
        )
    for r in (2, 8):
        out.append(
            (
                "A4",
                "fwd",
                f"r{r}",
                lambda r=r: TwoStageLocalizer(w, n_fwd=r, n_bwd=0, d2=dc, **kw),
            )
        )
    for d in ("fwd", "bidir"):
        for P in PS:
            for wt, wn in ((None, "dense"), (n // 4, "n4")):
                out.append(
                    (
                        "A5",
                        d,
                        f"P{P}{wn}",
                        lambda P=P, wt=wt, d=d: NonAdaptiveCSLocalizer(
                            w,
                            p_fwd=P if d == "fwd" else P // 2,
                            p_bwd=0 if d == "fwd" else P // 2,
                            weight=wt,
                            max_nnz=2 * k + 4,
                            **kw,
                        ),
                    )
                )
        for T in TS:
            for tag, th, tol, wt, df in A6_VARIANTS(n):
                out.append(
                    (
                        "A6",
                        d,
                        f"T{T}{tag}",
                        lambda T=T, th=th, tol=tol, wt=wt, df=df, d=d: GroupTestingLocalizer(
                            w,
                            t_fwd=T if d == "fwd" else T // 2,
                            t_bwd=0 if d == "fwd" else T // 2,
                            weight=wt,
                            thresh=th,
                            tol_frac=tol,
                            delta_frac=df,
                            **kw,
                        ),
                    )
                )
        for df, e in A7_VARIANTS:
            out.append(
                (
                    "A7",
                    d,
                    f"d{df}e{e}",
                    lambda df=df, e=e, d=d: AdaptiveBisectionLocalizer(
                        w, err=e, delta_frac=df, bidirectional=d == "bidir", **kw
                    ),
                )
            )
    for r in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        out.append(
            (
                "A3",
                "bidir",
                f"r{r}",
                lambda r=r: ResidualLocalizer(w, n_fwd=r, n_bwd=r, n_prune=16, **kw),
            )
        )
    for r in (1, 2, 4, 8, 16):
        out.append(
            (
                "A4",
                "bidir",
                f"r{r}",
                lambda r=r: TwoStageLocalizer(w, n_fwd=r, n_bwd=r, d2=dc, **kw),
            )
        )
    return out


# ------------------------------------------------------------------ worker


def _tile_state(point):
    t = point["tile_size"]
    spec = TileSpec(point["ckpt_seed"], point["layer"], point["r0"], point["c0"], t, t, t)
    W = _ckpt(point["ckpt_seed"])[point["layer"]]
    tile = Tile(spec, W[point["r0"] : point["r0"] + t, point["c0"] : point["c0"] + t].clone())
    return tile, encode(tile, EncodingSpec(**point["encoding"]))


def _inject(point, healthy, tile):
    fs = dict(point["fault"])
    fs["k"] = point["k"]
    gen = make_generator(
        point["fault_seed"] * 1000 + point["ckpt_seed"] + point["r0"] + point["c0"], "faults"
    )
    bad, truth = inject(healthy, FaultSpec(**fs), gen)
    wmax = float(tile.w.abs().max())
    return bad, truth.cells[truth.delta_w.abs() >= point["visible_frac"] * wmax]


def _seed(point):
    return point["fault_seed"] + 100003 * point["ckpt_seed"] + 1009 * point["r0"] + 31 * point["c0"]


def run_e1(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = _tile_state(point)
    bad, vis = _inject(point, healthy, tile)
    cur = [healthy]
    seed = _seed(point) + 7919 * point["k"]
    rows = []
    for sigma in point["sigmas"]:
        noise = ReadNoise(mode="per_cell", sigma_cell_frac=sigma, adc_bits=None, dac_bits=None)
        reads, f1s, npred = [], [], []
        for label, direction, cfg, build in designs(tile.w, point["k"], point["fa"], seed):
            reader = TileReader(
                lambda: cur[0],
                noise,
                make_generator(seed, f"read_{label}_{direction}_{cfg}"),
                backward_supported=direction == "bidir",
            )
            loc = build()
            cur[0] = healthy
            loc.calibrate(reader)
            cur[0] = bad
            est, cost = loc.localize(reader)
            reads.append(cost.mvm_fwd + cost.mvm_bwd)
            f1s.append(round(localization_f1(est.cells, vis), 5))
            npred.append(len(est.cells))
        rows.append(
            {
                "sigma_c": sigma,
                "n_visible": len(vis),
                "reads": json.dumps(reads),
                "f1": json.dumps(f1s),
                "n_pred": json.dumps(npred),
            }
        )
    return rows


# ------------------------------------------------------------------ E1 summary


def _best(reads, f1, mask, target):
    """(mean-reads, config index, mean f1) of the cheapest config in mask with mean F1>=target."""
    ok = mask & (f1 >= target)
    if not ok.any():
        return math.inf, -1, float(f1[mask].max()) if mask.any() else float("nan")
    idx = np.flatnonzero(ok)[np.argmin(reads[ok])]
    return float(reads[idx]), int(idx), float(f1[idx])


def summarize_e1(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    target, rng = cfg["target_f1"], np.random.default_rng(0)
    cat = {}
    verdict_rows, design_rows = [], []
    for (k, sigma, tsz), g in raw.groupby(["k", "sigma_c", "tile_size"]):
        if (k, tsz) not in cat:
            dummy = torch.zeros(tsz, tsz, dtype=DTYPE)
            cat[(k, tsz)] = [(a, b, c) for a, b, c, _ in designs(dummy, int(k), 0.1, 0)]
        names = cat[(k, tsz)]
        R = np.array([json.loads(s) for s in g.reads], dtype=float)  # (I, C)
        F = np.array([json.loads(s) for s in g.f1], dtype=float)
        algo = np.array([a for a, _, _ in names])
        dirn = np.array([d for _, d, _ in names])
        fm, rm = F.mean(0), R.mean(0)
        fwd, bid = dirn == "fwd", dirn == "bidir"
        rf, jf, _ = _best(rm, fm, fwd, target)
        rb, jb, f1b = _best(rm, fm, bid, target)
        ra, _, _ = _best(rm, fm, fwd | bid, target)
        # paired bootstrap over instances of the ratio of best-reads
        I = len(R)
        ratios = np.empty(cfg["n_boot"])
        for b in range(cfg["n_boot"]):
            ix = rng.integers(0, I, I)
            fb, rbm = F[ix].mean(0), R[ix].mean(0)
            x, _, _ = _best(rbm, fb, bid, target)
            y, _, _ = _best(rbm, fb, fwd, target)
            ratios[b] = (
                np.nan
                if math.isinf(x) and math.isinf(y)
                else math.inf
                if math.isinf(x)
                else 0.0
                if math.isinf(y)
                else x / y
            )
        ratio = (
            np.nan
            if math.isinf(rb) and math.isinf(rf)
            else math.inf
            if math.isinf(rb)
            else 0.0
            if math.isinf(rf)
            else rb / rf
        )
        hi = float(np.nanquantile(ratios, 0.975)) if np.isfinite(ratios).any() else np.nan
        per = {}
        for a in ("A1", "A2", "A2b", "A3", "A4", "A5", "A6", "A7"):
            for d in ("fwd", "bidir"):
                mk = (algo == a) & (dirn == d)
                if mk.any():
                    per[f"reads09__{a}_{d}"] = _best(rm, fm, mk, target)[0]
        ok = bool(f1b >= target and ratio <= 0.7 and hi <= 0.8) if not math.isnan(ratio) else False
        a2 = per.get("reads09__A2_fwd", math.inf)
        a4 = per.get("reads09__A4_bidir", math.inf)
        verdict_rows.append(
            {
                "experiment": "E1",
                "equal_area": True,
                "k": int(k),
                "sigma_c": float(sigma),
                "tile_size": int(tsz),
                "n_instances": I,
                "reads_fwd_best": rf,
                "reads_bidir_best": rb,
                "best_fwd_cfg": f"{algo[jf]}:{names[jf][2]}" if jf >= 0 else "none",
                "best_bidir_cfg": f"{algo[jb]}:{names[jb][2]}" if jb >= 0 else "none",
                "cell_f1_bidir": f1b,
                "reads_ratio": ratio,
                "reads_ratio_ci_upper": hi,
                "gain_G": (rf / rb) if rb > 0 and not math.isinf(rf) else np.nan,
                "c1_row_ok": int(ok),
                "reads_to_f1_0p9": ra,
                "reads_to_f1_0p9_fwd": rf,
                "reads_to_f1_0p9_bidir": rb,
                "a2_vs_a4_reads_rel_diff": (abs(a2 - a4) / a4)
                if k == 1 and a4 > 0 and not (math.isinf(a2) or math.isinf(a4))
                else np.nan,
                **per,
            }
        )
        for j, (a, d, c) in enumerate(names):
            design_rows.append(
                {
                    "k": int(k),
                    "sigma_c": float(sigma),
                    "tile_size": int(tsz),
                    "algo": a,
                    "direction": d,
                    "cfg": c,
                    "reads": rm[j],
                    "f1": fm[j],
                }
            )
    return pd.DataFrame(verdict_rows), pd.DataFrame(design_rows)


# ------------------------------------------------------------------ E1b


def _Q(x):
    return norm.sf(x)


def _h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def predict_w_star(scheme, n, delta, z_np, sd_fn, grid):
    """T3-style prediction: the weight maximizing the per-test information about WHICH of n items
    of a line is faulty (s = 1 fault per line), for a test of weight w with output noise sd_fn(w).
    scheme "group": 0/1 probe, 1-bit output positive when |r| > z_np*sd(w) (binary channel with
    miss / false-alarm probabilities). scheme "cs": +-1 random-sign probe, ternary input
    {-1,0,+1} -> real output, mutual information by numerical integration."""
    best, best_i = None, -1.0
    for w in grid:
        p, s = w / n, sd_fn(w)
        thr = z_np * s
        if scheme == "group":
            miss = norm.cdf((thr - delta) / s) - norm.cdf((-thr - delta) / s)
            fa = 2 * _Q(thr / s)
            py = p * (1 - miss) + (1 - p) * fa
            info = _h2(py) - p * _h2(miss) - (1 - p) * _h2(fa)
        else:
            y = np.linspace(-delta - 8 * s, delta + 8 * s, 4001)
            px = {-1: p / 2, 0: 1 - p, 1: p / 2}
            dens = {x: norm.pdf(y, x * delta, s) for x in px}
            mix = sum(px[x] * dens[x] for x in px)
            hy = -trapezoid(mix * np.log2(np.maximum(mix, 1e-300)), y)
            hn = 0.5 * np.log2(2 * np.pi * np.e * s * s)
            info = hy - hn
        if info > best_i:
            best, best_i = w, info
    return float(best)


class FixedOutputNoiseReader:
    """TileReader wrapper adding probe-independent output noise N(0, s_out^2) per output and read
    (TIA / ADC floor): the output-referred noise model of C2."""

    def __init__(self, inner: TileReader, s_out: float, gen: torch.Generator) -> None:
        self._in, self.s_out, self.gen = inner, s_out, gen
        self.backward_supported = inner.backward_supported

    cost = property(lambda self: self._in.cost)
    shape = property(lambda self: self._in.shape)

    def mvm(self, x):
        y = self._in.mvm(x)
        return y + self.s_out * torch.randn(*y.shape, generator=self.gen, dtype=DTYPE)

    def mvm_t(self, y):
        z = self._in.mvm_t(y)
        return z + self.s_out * torch.randn(*z.shape, generator=self.gen, dtype=DTYPE)

    def full_readout(self):
        return self._in.full_readout()


def run_e1b(point: dict) -> list[dict]:
    torch.set_num_threads(1)
    tile, healthy = _tile_state(point)
    bad, vis = _inject(point, healthy, tile)
    n = tile.w.shape[1]
    wmax = float(tile.w.abs().max())
    seed = _seed(point) + 7919 * point["k"]
    rows = []
    cur = [healthy]
    for sigma in point["sigmas"]:
        for model in ("per_cell", "output_fixed"):
            noise = ReadNoise(
                mode="per_cell",
                sigma_cell_frac=sigma if model == "per_cell" else 0.0,
                adc_bits=None,
                dac_bits=None,
            )
            algos = ("A6", "A5") if model == "per_cell" else ("A6",)
            for algo in algos:
                for w in point["weights"]:
                    for T in point["budgets"]:
                        gen = make_generator(seed, f"read_{model}_{algo}_{w}_{T}")
                        inner = TileReader(lambda: cur[0], noise, gen)
                        reader = (
                            inner
                            if model == "per_cell"
                            else FixedOutputNoiseReader(
                                inner, sigma * wmax * math.sqrt(n), make_generator(seed, f"o{w}{T}")
                            )
                        )
                        kw = {"fa": point["fa"], "seed": seed}
                        if algo == "A5":
                            loc = NonAdaptiveCSLocalizer(tile.w, p_fwd=T, weight=w, **kw)
                        else:
                            loc = GroupTestingLocalizer(
                                tile.w,
                                t_fwd=T,
                                weight=w,
                                thresh="np",
                                noise_model="cell" if model == "per_cell" else "floor",
                                **kw,
                            )
                        cur[0] = healthy
                        loc.calibrate(reader)
                        cur[0] = bad
                        est, cost = loc.localize(reader)
                        rows.append(
                            {
                                "sigma_c": sigma,
                                "model": model,
                                "algo": algo,
                                "weight": w,
                                "budget": T,
                                "reads": cost.mvm_fwd + cost.mvm_bwd,
                                "f1": localization_f1(est.cells, vis),
                                "delta_med": float(
                                    (bad.weights() - healthy.weights())[vis[:, 0], vis[:, 1]]
                                    .abs()
                                    .median()
                                    / wmax
                                )
                                if len(vis)
                                else float("nan"),
                            }
                        )
    return rows


def summarize_e1b(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = cfg["target_f1"]
    n = cfg["tile_size"]
    # units of max|w|: actual median visible-fault deviation; NP threshold z * sd(w), z ~ 4.5
    delta_act = float(raw.delta_med.median())
    z_np = 4.5
    curve = (
        raw.groupby(["sigma_c", "model", "algo", "weight", "budget"])
        .agg(f1=("f1", "mean"), reads=("reads", "mean"))
        .reset_index()
    )
    out, grid = [], np.geomspace(2, n, 120)
    for (sigma, model, algo), g in curve.groupby(["sigma_c", "model", "algo"]):
        r09 = {}
        for w, gw in g.groupby("weight"):
            ok = gw[gw.f1 >= target]
            r09[w] = float(ok.reads.min()) if len(ok) else math.inf
        fin = {w: r for w, r in r09.items() if math.isfinite(r)}
        if fin:
            rmin = min(fin.values())
            tied = [w for w, r in fin.items() if r <= rmin * 1.0001]
            w_meas = float(np.exp(np.mean(np.log(tied))))
        else:
            w_meas = np.nan
        # prediction in units of max|w|
        # (units of max|w|): per-cell sd = sigma*sqrt(w), fixed floor sd = sigma*sqrt(n)
        sd_fn = (
            (lambda w, s=sigma: s * math.sqrt(w))
            if model == "per_cell"
            else (lambda w, s=sigma: s * math.sqrt(n))
        )
        w_pred = predict_w_star("cs" if algo == "A5" else "group", n, delta_act, z_np, sd_fn, grid)
        out.append(
            {
                "experiment": "E1b",
                "sigma_c": sigma,
                "noise_model": model,
                "algo": algo,
                "w_meas": w_meas,
                "w_pred": w_pred,
                "reads09_at_w": json.dumps({int(k): v for k, v in r09.items()}),
                "optimal_weight_meas_over_pred": (w_meas / w_pred) if algo == "A6" else np.nan,
                "optimal_weight_meas_over_pred_a5": (w_meas / w_pred) if algo == "A5" else np.nan,
            }
        )
    return pd.DataFrame(out), curve


# ------------------------------------------------------------------ driver


def build_points(cfg, sub=None):
    s = {**cfg, **(sub or {})}
    pts = []
    t = s["tile_size"]
    for cs in s["ckpt_seeds"]:
        for layer in s["layers"]:
            M, N = load_checkpoint(cs)[layer].shape
            origins = [(r, c) for r in range(0, M - t + 1, t) for c in range(0, N - t + 1, t)]
            for r0, c0 in origins[: s["tiles_per_layer"]]:
                for fseed in range(s["fault_seeds"]):
                    for k in s["ks"]:
                        pts.append(
                            {
                                "experiment": cfg["experiment"],
                                "ckpt_seed": cs,
                                "layer": layer,
                                "r0": r0,
                                "c0": c0,
                                "tile_size": t,
                                "fault_seed": fseed,
                                "k": k,
                                "sigmas": s["sigmas"],
                                "encoding": cfg["encoding"],
                                "fault": cfg["fault"],
                                "visible_frac": cfg["visible_frac"],
                                "fa": cfg["fa"],
                                **(
                                    {
                                        "weights": cfg["weights"],
                                        "budgets": cfg["budgets"],
                                    }
                                    if cfg["experiment"] == "e1b"
                                    else {}
                                ),
                            }
                        )
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "e1b"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load((CFG_DIR / f"b_{a.exp}.yaml").read_text())
    smoke = cfg.pop("smoke")
    if a.smoke:
        cfg.update(smoke)
    tag = "smoke" if a.smoke else "full"
    out = Path(a.out or REPO / "results" / "spectral" / f"b_{a.exp}")
    pts = build_points(cfg)
    for sub in cfg.get("subsets", []):
        pts += build_points(cfg, sub)
    print(f"{a.exp} {tag}: {len(pts)} points", flush=True)
    fn = run_e1 if a.exp == "e1" else run_e1b
    raw = run_grid(fn, pts, jobs=a.jobs, out_dir=out / f"_cells_{tag}")
    if a.exp == "e1":
        vm, designs_df = summarize_e1(raw, cfg)
        metrics = vm
    else:
        vm, designs_df = summarize_e1b(raw, cfg)
        metrics = vm
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(vm.drop(columns=[c for c in vm.columns if c.startswith("reads09__")]).to_string())
    rd = write_run(
        out,
        cfg,
        0,
        metrics,
        params={"visible_frac": cfg["visible_frac"], "fa": cfg["fa"]},
        extra={"mode": tag, "n_points": len(pts)},
        allow_dirty=a.allow_dirty,
        run_id=None,
    )
    metrics.to_csv(rd / "verdict_metrics.csv", index=False)
    designs_df.to_csv(rd / "designs_summary.csv", index=False)
    print("wrote", rd)


if __name__ == "__main__":
    main()

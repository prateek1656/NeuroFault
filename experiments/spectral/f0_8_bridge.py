# ruff: noqa: C408
"""F0.8 network-bridge validation (9; criteria/f0_validity.yaml F0.8a-c) + the
report-only a_bnfold M4-vs-B8 re-check on the CrossSim tier.

Tier 1 = digital twin (Exp2Bridge: effective conv matrices + mean-bias into the float model).
Tier 2 = the SAME matrices/biases pushed through accessor.write into the repo's CrossSim system
         (8-bit DAC, calibrated 8-bit ADC; set_matrix adds no device error, see spectral/bridge.py).
Tier 3 = AIHWKit-native PCM drift + GlobalDriftCompensation (ideal forward) vs our D2 + GDC (B1b).

The twin side reuses a_bnfold's machinery unchanged (same streams: ckpt_seed + 1000, same 8f+8b M4,
same B8 selection), so folded D2 cells are the a_bnfold cells evaluated on the first 5000 test
images. Metrics emitted with the frozen names: f08a_tier1_vs_tier2_abs_diff_pt,
f08b_ours_vs_aihwkit_abs_diff_pt, f08c_abs_dev_over_binomial_halfwidth.

  python experiments/spectral/f0_8_bridge.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # aihwkit + torch both ship libomp
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import a_bnfold as ab
import a_common as ac
import numpy as np
import pandas as pd
import torch
from a_common import BN_OF, CONV_LAYERS, INCUMBENTS, TileCtx, best_of, make_drift
from a_exp2_jobs import Exp2Bridge
from f0_common import finish, get_tile, load_config, parse_args

from neurofault.spectral.baselines import GDC, bind_all
from neurofault.spectral.bnfold import SEED_OFFSET, tile_origins
from neurofault.spectral.bridge import TierTwo, aihwkit_native, f08_metric_rows
from neurofault.spectral.drift import PCMPerDevice
from neurofault.spectral.encoding import encode
from neurofault.spectral.exp2 import cluster_bootstrap_ci
from neurofault.spectral.harness import run_grid
from neurofault.spectral.noise import program
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import ReadNoise, make_generator


def _times_for(cfg: dict, seed: int, mode: str) -> dict:
    if mode == "folded":
        return dict(cfg["folded_times"])
    t = dict(cfg["std_times"])
    if seed == 42:
        t.update(cfg.get("std_extra_times_seed42", {}))
    return t


def _bridge(seed: int, mode: str, n_eval: int) -> Exp2Bridge:
    b = Exp2Bridge(seed, "W2" if mode == "folded" else "W1", n_eval, cal_n=64)
    b._seed_key = seed
    return b


def _ckpt_stored(seed: int) -> float:
    from neurofault.spectral.tiles import checkpoint_path

    blob = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    return float(blob["final_test_acc"])


def _scatter(store: dict, layer: str, shape, r0, c0, m, n, w) -> None:
    if layer not in store:
        store[layer] = torch.zeros(shape, dtype=torch.float32)
    store[layer][r0 : r0 + m, c0 : c0 + n] = w.float()


# --------------------------------------------------------------------------- tiers 1 + 2


def t12_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, seed, mode, sc = p["cfg"], p["seed"], p["mode"], p["scenario"]
    bridge = _bridge(seed, mode, cfg["n_eval"])
    bn_of = {k: v for k, v in BN_OF.items()}
    t2 = TierTwo(bridge.model, bn_of, seed)  # copies the model BEFORE any bridge.acc() call
    saved = ab._install("cifar10", seed, bridge)
    rseed = seed + SEED_OFFSET
    stored = _ckpt_stored(seed)
    rows: list[dict] = []
    base = dict(seed=seed, mode=mode, scenario=sc["name"], n_eval=cfg["n_eval"], stored=stored)

    def both(arm, tlabel, t, mats, bias):
        a1 = bridge.acc(mats, bias)
        t2.set_state(mats, bias)
        a2 = t2.acc(bridge.xe, bridge.ye)
        rows.append(dict(**base, arm=arm, time=tlabel, t_s=t, acc_t1=a1, acc_t2=a2))

    try:
        mats = bridge.layer_mats()
        shapes = {k: tuple(v.shape) for k, v in mats.items()}
        mod = make_drift(sc["model"], sc.get("field") or "row_gradient", sc["rho"], 1.0)
        runs, wp, w0 = [], {}, {}
        for layer, r0, c0, m, n in ab._tiles(bridge):
            ctx = TileCtx(seed, layer, r0, c0, m, n, 256)
            real = mod.sample(ctx.stp, make_generator(rseed, f"drift:{layer}:{r0}:{c0}"))
            _scatter(wp, layer, shapes[layer], r0, c0, m, n, ctx.wp)
            _scatter(w0, layer, shapes[layer], r0, c0, m, n, ctx.st0.weights())
            run = ab.BFRun(ctx.slim(), cfg["sigma"], rseed)
            run.real = real
            runs.append(((layer, r0, c0, m, n), run))
        if sc["name"] == "D2":  # scenario-independent reference points: once per (seed, mode)
            both("digital", "t0", 0.0, None, None)
            both("pipeline", "t0", 0.0, w0, None)  # encode -> decode -> reassemble, no noise
            both("prog", "t0", 0.0, wp, None)
        for tlabel, t in _times_for(cfg, seed, mode).items():
            arms = cfg["folded_arms" if mode == "folded" else "std_arms"]
            W = {a: {} for a in arms}
            B = {a: {} for a in arms}
            for (layer, r0, c0, m, n), run in runs:
                res = run.at(t)
                for a in arms:
                    key = {"none": "none", "b8": best_of(res, INCUMBENTS), "m4": "m4"}[a]
                    r = res[key]
                    w = res["_wt"] if key == "none" else r["W"]
                    _scatter(W[a], layer, shapes[layer], r0, c0, m, n, w)
                    if key != "none":
                        bd = B[a].setdefault(layer, torch.zeros(shapes[layer][0]))
                        bd[r0 : r0 + m] += r["b"].float()
            for a in arms:
                both(a, tlabel, t, W[a], B[a] or None)
            del W, B
        return rows
    finally:
        ab._restore(saved)


# --------------------------------------------------------------------------- tier 3


def _eval(model, bridge) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(bridge.xe), 500):
            correct += int(
                (model(bridge.xe[i : i + 500]).argmax(1) == bridge.ye[i : i + 500]).sum()
            )
    return 100.0 * correct / len(bridge.xe)


def t3_job(p: dict) -> list[dict]:
    """Our D2 + GDC(aihwkit variant, noise-free identity reads) vs AIHWKit-native D2 + GDC, standard
    BN network, R realizations each, accuracy averaged over realizations (the realization-to-
    realization spread is sampling noise, not a model difference; per-realization values are kept)."""
    torch.set_num_threads(1)
    cfg, seed = p["cfg"], p["seed"]
    n_real = cfg["t3_realizations"]
    times = dict(cfg["t3_times"])
    bridge = _bridge(seed, "std", cfg["n_eval"])
    base_model = bridge.model
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    ckm = bridge.layer_mats()
    shapes = {k: tuple(v.shape) for k, v in ckm.items()}
    tiles = tile_origins(ckm, CONV_LAYERS, 256, 16)
    quiet = ReadNoise("per_cell", 0.0, adc_bits=None, dac_bits=None)
    ours = {}  # (r, tlabel) -> acc ; tlabel "prog" for the undrifted programmed state
    for r in range(n_real):
        gp = make_generator(seed * 100 + r, "t3:prog")
        runs = []
        for layer, r0, c0, m, n in tiles:
            tl = get_tile(seed, layer, r0, c0, m, n, 256)
            stp = program(
                encode(tl, ac.ENC),
                ac.PROG,
                make_generator(seed * 100 + r, f"t3:prog:{layer}:{r0}:{c0}"),
            )
            real = PCMPerDevice().sample(
                stp, make_generator(seed * 100 + r, f"t3:drift:{layer}:{r0}:{c0}")
            )
            cur = [stp]
            reader = TileReader(lambda c=cur: c[0], quiet, gp)
            g = GDC("aihwkit")
            bind_all([g], lambda c=cur: c[0].weights(), lambda t_=tl: t_.w)
            g.calibrate(reader)
            runs.append(((layer, r0, c0, m, n), stp, real, cur, reader, g))
        wp = {}
        for (layer, r0, c0, m, n), stp, *_ in runs:
            _scatter(wp, layer, shapes[layer], r0, c0, m, n, stp.weights())
        ours[(r, "prog")] = bridge.acc(wp)
        for tlabel, t in times.items():
            W = {}
            for (layer, r0, c0, m, n), _stp, real, cur, reader, g in runs:
                cur[0] = real.at(t)
                _scatter(W, layer, shapes[layer], r0, c0, m, n, g.update(reader, t).W_eff)
            ours[(r, tlabel)] = bridge.acc(W)
        del runs
    rows = []
    for variant, rn in (("as_is", 1.0), ("no_1f", 0.0)):
        nat = {}
        for r in range(n_real):
            base_model.load_state_dict(base_sd)
            am = aihwkit_native(base_model, read_noise_scale=rn, seed=seed * 100 + r)
            nat[(r, "prog")] = _eval(am, bridge)
            for tlabel, t in times.items():
                am.drift_analog_weights(t)
                nat[(r, tlabel)] = _eval(am, bridge)
            del am
        for tlabel in ["prog", *times]:
            o = [ours[(r, tlabel)] for r in range(n_real)]
            a = [nat[(r, tlabel)] for r in range(n_real)]
            rows.append(dict(seed=seed, variant=variant, time=tlabel,
                             t_s=0.0 if tlabel == "prog" else times[tlabel],
                             n_real=n_real, n_eval=cfg["n_eval"],
                             acc_ours=float(np.mean(o)), acc_aihwkit=float(np.mean(a)),
                             acc_ours_per_real=o, acc_aihwkit_per_real=a))  # fmt: skip
    return rows


# --------------------------------------------------------------------------- assembly


def headline_table(t12: pd.DataFrame) -> pd.DataFrame:
    """Report-only: M4 - B8 on the BN-folded network, tier 1 vs tier 2, pooled over scenarios
    (same pooling and paired cluster-bootstrap lower bound as a_bnfold / BF2)."""
    f = t12[(t12["mode"] == "folded") & t12.arm.isin(["b8", "m4"])]
    if f.empty:
        return pd.DataFrame()
    pv = f.pivot_table(index=["seed", "scenario", "time", "t_s"], columns="arm",
                       values=["acc_t1", "acc_t2"]).reset_index()  # fmt: skip
    pv.columns = ["_".join(c).strip("_") for c in pv.columns]
    if "acc_t1_b8" not in pv or "acc_t1_m4" not in pv:
        return pd.DataFrame()
    out = []
    for t_s, g in pv.groupby("t_s"):
        row = dict(t_s=t_s, n_rows=len(g))
        for tier in ("t1", "t2"):
            gain = (g[f"acc_{tier}_m4"] - g[f"acc_{tier}_b8"]).to_numpy()
            mean, lo, _hi = cluster_bootstrap_ci(gain, g.seed.to_numpy(), 4000, 0.05, 1)
            row.update({f"acc_b8_{tier}": g[f"acc_{tier}_b8"].mean(),
                        f"acc_m4_{tier}": g[f"acc_{tier}_m4"].mean(),
                        f"gain_{tier}": mean, f"gain_ci_lower_{tier}": lo})  # fmt: skip
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "f0_8.yaml", "f0_8", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed, seed_offset=SEED_OFFSET)
    out = Path(a.out)
    pts12 = [dict(cfg=cfg, seed=s, mode=m, scenario=sc)
             for sc in cfg["scenarios"] for m in ("std", "folded") for s in cfg["ckpt_seeds"]]  # fmt: skip
    pts3 = [dict(cfg=cfg, seed=s) for s in cfg["ckpt_seeds"]]
    t12 = run_grid(t12_job, pts12, jobs=a.jobs, out_dir=out / "_cells_t12")
    t3 = run_grid(t3_job, pts3, jobs=a.jobs, out_dir=out / "_cells_t3")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    n_eval = cfg["n_eval"]
    drifted = t12[t12.arm.isin(["none", "b8", "m4"])]
    # F0.8a, strict scope: every drifted matched cell (all seeds, all times, both modes)
    a_rows = f08_metric_rows(drifted.to_dict("records"), [], [], n_eval)
    # F0.8b: official = AIHWKit as shipped (with its 1/f read noise), drifted times only
    t3o = t3[(t3.variant == "as_is") & (t3.time != "prog")]
    b_rows = f08_metric_rows([], t3o.drop(columns=["acc_ours_per_real", "acc_aihwkit_per_real"]).to_dict("records"), [], n_eval)  # fmt: skip
    # F0.8c: no-drift tier-1 pipeline (encode->decode->reassemble) vs stored final_test_acc, std net
    c_rows = f08_metric_rows([], [], t12[(t12.arm == "pipeline") & (t12["mode"] == "std")].to_dict("records"), n_eval)  # fmt: skip
    metrics = pd.DataFrame(a_rows + b_rows + c_rows)
    is_a = metrics.f08a_tier1_vs_tier2_abs_diff_pt.notna()
    reg = metrics[
        ~is_a | ((metrics.seed == 42) & metrics.t_s.isin([86400.0, 2592000.0, 31557600.0]))
    ]
    t12["diff_pt"] = (t12.acc_t1 - t12.acc_t2).abs()
    print("== tier-1 vs tier-2 (all cells) ==")
    print(t12.drop(columns=["stored", "n_eval"]).round(3).to_string(index=False))
    print("== tier 3 ==")
    t3["diff_pt"] = (t3.acc_ours - t3.acc_aihwkit).abs()
    print(
        t3.drop(columns=["acc_ours_per_real", "acc_aihwkit_per_real"])
        .round(3)
        .to_string(index=False)
    )
    hl = headline_table(t12)
    print("== a_bnfold headline on the CrossSim tier (BN-folded, M4 - B8) ==")
    print(hl.round(3).to_string(index=False))
    finish("F0.8 bridge", "f0_validity.yaml", cfg, a, metrics,
                        extra={"note": "F0.0 rows absent by design (separate script)",
                               "f08a_scope": "strict: all drifted matched cells"},
                        side_tables=dict(tier12_raw=t12, tier3_raw=t3, headline_crosssim=hl,
                                         metrics_registered_scope_f08a=reg))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

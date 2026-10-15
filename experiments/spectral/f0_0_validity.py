# ruff: noqa: C408
"""F0.0 validity gates (F0.0a-e): nothing else is trusted until these pass.

  a  probe reproduction  - the original feasibility probe, re-implemented on the new stack
  b  D2 vs AIHWKit       - PCMLikeNoiseModel distributions (drift part, 1/f part separately)
  c  GDC parity          - mean-|readout| alpha (B1b formula, implemented locally here so this
                           script does not depend on baselines/) vs aihwkit GlobalDriftCompensation
  d  D0 parity           - LegacyRailShift vs injection._drift through tests/unit/fakes.make_handle
  e  persistence / invariants

Metric column names are the contract with experiments/spectral/criteria/f0_validity.yaml (frozen;
F0.8a-c need the network bridge and are out of scope here, so they stay INCONCLUSIVE and the
overall verdict of this script is INCONCLUSIVE by construction - read the F0.0a-e rows).

  python experiments/spectral/f0_0_validity.py --jobs 4 --allow-dirty
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from f0_common import (
    PROBE_LAYERS,
    EncodingSpec,
    FaultSpec,
    ReadNoise,
    TileReader,
    ckpt_mats,
    enc_from_label,
    encode,
    finish,
    get_tile,
    inject,
    load_config,
    make_generator,
    parse_args,
    run_old_probe,
)
from scipy import stats

from neurofault.spectral.harness import run_grid
from neurofault.spectral.noise import read_noise_cell
from neurofault.spectral.types import DTYPE, T0_S, ConductanceState

# tabulated reference numbers
REF_SIGMA1_SNR = (0.05, 0.09)
REF_PROBE_SNR = 3.1
REF_ROWLOC = (0.55, 0.62)


def range_rel_err(x: float, lo: float, hi: float) -> float:
    """0 inside [lo, hi]; distance to the nearest edge divided by that edge outside."""
    if x < lo:
        return (lo - x) / lo
    if x > hi:
        return (x - hi) / hi
    return 0.0


# --------------------------------------------------------------------------- F0.0a


def check_a(p: dict) -> list[dict]:
    seed, layer, ck = p["seed"], p["layer"], p["ckpt_seed"]
    r = run_old_probe(ck, layer, seed, T=p["T"], trials=p["trials"], p_multi=8)
    # gap reference: the original probe's own computation (float32 SVD of W[:T,:T])
    W = ckpt_mats(ck)[layer].float()[: p["T"], : p["T"]]
    S = torch.linalg.svdvals(W)
    gap_ref = float(S[0] / S[1])
    return [
        {
            "check": "F0.0a",
            "layer": layer,
            "sigma1_snr_med": r["sigma1_snr_med"],
            "probe_snr_med": r["probe_snr_med"],
            "rowloc_p1": r["rowloc_p1"],
            "gap_new": r["gap"],
            "gap_orig": gap_ref,
            "sigma1_det3": r["sigma1_det3"],
            "f00a_sigma1_snr_rel_err": range_rel_err(r["sigma1_snr_med"], *REF_SIGMA1_SNR),
            "f00a_probe_snr_rel_err": abs(r["probe_snr_med"] - REF_PROBE_SNR) / REF_PROBE_SNR,
            "f00a_rowloc_rel_err": range_rel_err(r["rowloc_p1"], *REF_ROWLOC),
            "f00a_gap_abs_err": abs(r["gap"] - gap_ref),
        }
    ]


# --------------------------------------------------------------------------- F0.0b


def check_b(p: dict) -> list[dict]:
    from aihwkit.inference.noise.pcm import PCMLikeNoiseModel

    from neurofault.spectral.drift import PCMPerDevice

    N, gmax = p["n_devices"], 25.0
    level = p["level_frac"] * gmax
    model = PCMLikeNoiseModel(g_max=gmax)
    torch.manual_seed(int(1e6 * p["level_frac"]) + p["seed"])  # aihwkit uses the global RNG
    enc = EncodingSpec("diff_pair", g_min_frac=0.0)
    g0 = torch.full((N, 1), level, dtype=DTYPE)
    st0 = ConductanceState(g0.clone(), torch.zeros_like(g0), 1.0, enc)
    gen = make_generator(p["seed"], f"f00b:{p['level_frac']}")

    rows = []

    def cmp(part, t, ours, ref, **kw):
        ours, ref = ours.double().flatten(), ref.double().flatten()
        ks = stats.ks_2samp(ours.numpy(), ref.numpy())
        rows.append(
            {
                "check": "F0.0b",
                "part": part,
                "level_frac": p["level_frac"],
                "t_s": t,
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "mean_ours": float(ours.mean()),
                "mean_ref": float(ref.mean()),
                "std_ours": float(ours.std()),
                "std_ref": float(ref.std()),
                **kw,
            }
        )
        return rows[-1]

    def finalize(row, informational=False):
        row["mean_ratio_abs_err"] = abs(row["mean_ours"] / row["mean_ref"] - 1.0)
        if not informational:
            row["f00b_ks_p"] = row["ks_p"]
            row["f00b_mean_ratio_abs_err"] = row["mean_ratio_abs_err"]
        else:
            row["f00b_info_ks_p"] = row["ks_p"]

    # drift part: nu, and G(t)/G0 with the 1/f noise switched off in aihwkit
    nu_ref = model.generate_drift_coefficients(torch.full((N,), level, dtype=DTYPE))
    real = PCMPerDevice().sample(st0, gen)
    nu_ours = real.nu[:, 0, 0]
    finalize(cmp("nu", 0.0, nu_ours, nu_ref))
    model_nonoise = PCMLikeNoiseModel(g_max=gmax, read_noise_scale=0.0)
    onef = ReadNoise("output_referred", 0.0, 0.0, one_over_f=True, adc_bits=None, dac_bits=None)
    for t in p["times_s"]:
        ratio_ours = real.at(t).g_pos[:, 0] / level
        ref_g = model_nonoise.apply_drift_noise_to_conductance(
            torch.full((N,), level, dtype=DTYPE), nu_ref, t
        )
        finalize(cmp("drift", t, ratio_ours, ref_g / level))
        # 1/f part alone: nu = 0 in aihwkit, one_over_f read noise on the undrifted tile in ours
        ref_f = model.apply_drift_noise_to_conductance(
            torch.full((N,), level, dtype=DTYPE), torch.zeros(N, dtype=DTYPE), t
        )
        w_f = read_noise_cell(st0, onef, t, gen)
        finalize(cmp("onef", t, w_f.flatten() / (level / st0.scale), ref_f / level))
        # informational: drift then 1/f in sequence (q_s evaluated on the drifted g in ours, on
        # the programmed g in aihwkit - a documented implementation difference, not a criterion)
        comb_ref = model.apply_drift_noise_to_conductance(
            torch.full((N,), level, dtype=DTYPE), nu_ref, t
        )
        st_t = real.at(t)
        comb_ours = read_noise_cell(st_t, onef, t, gen)
        finalize(
            cmp("drift+onef", t, comb_ours.flatten() * st0.scale / level, comb_ref / level),
            informational=True,
        )
    return rows


# --------------------------------------------------------------------------- F0.0c


def gdc_alpha_ours(reader0: TileReader, readert: TileReader, n: int) -> float:
    """B1b formula: alpha = mean|y_ref| / mean|y_t| from one-hot MVMs (n of them). Same
    estimator AIHWKit's GlobalDriftCompensation uses, but read through our counted TileReader."""
    eye = torch.eye(n, dtype=DTYPE)
    ref = reader0.mvm(eye).abs().mean().clamp(min=1e-4)
    cur = readert.mvm(eye).abs().mean().clamp(min=1e-4)
    return float(ref / cur)


def check_c(p: dict) -> list[dict]:
    from aihwkit.inference.compensation.drift import GlobalDriftCompensation

    from neurofault.spectral.drift import PCMPerDevice

    ck, layer, T, sig = p["ckpt_seed"], p["layer"], p["T"], p["noise_frac"]
    tile = get_tile(ck, layer, 0, 0, T, T, T)
    st0 = encode(tile, EncodingSpec("diff_pair", g_min_frac=0.0))
    real = PCMPerDevice().sample(st0, make_generator(p["seed"], f"f00c:{ck}:{layer}"))
    gdc = GlobalDriftCompensation()
    n = st0.shape[1]
    eye = torch.eye(n, dtype=DTYPE)
    W0 = st0.weights()
    rows = []
    for t in p["times_s"]:
        stt = real.at(t)
        Wt = stt.weights()
        # noise-free: ours through a noise-free reader, native through the aihwkit class
        quiet = ReadNoise("per_cell", 0.0, adc_bits=None, dac_bits=None)
        g = make_generator(p["seed"], "f00c:quiet")
        a_ours = gdc_alpha_ours(
            TileReader(lambda: st0, quiet, g), TileReader(lambda s=stt: s, quiet, g), n
        )
        a_nat = float(gdc.apply(Wt @ eye, gdc.readout(W0 @ eye)))
        rows.append(
            {
                "check": "F0.0c",
                "ckpt_seed": ck,
                "layer": layer,
                "t_s": t,
                "mode": "noise_free",
                "alpha_ours": a_ours,
                "alpha_native": a_nat,
                "f00c_rel_diff_noise_free": abs(a_ours - a_nat) / a_nat,
            }
        )
        # noisy: independent per-cell read noise in the two estimators. Bound fixed from the
        # noise model BEFORE looking at the diff: alpha = S0/St, S = mean|W + eps| over m*n
        # cells has relative std <= sigma/(sqrt(mn) S); two independent alphas, each with a
        # noisy reference and a noisy current read, differ by std = sqrt(2)*sqrt(rel0^2+relt^2);
        # bound = 5 * that.
        m_, n_ = Wt.shape
        s0, st_ = sig * float(W0.abs().max()), sig * float(Wt.abs().max())
        rel0 = s0 / (math.sqrt(m_ * n_) * float(W0.abs().mean()))
        relt = st_ / (math.sqrt(m_ * n_) * float(Wt.abs().mean()))
        bound = 5.0 * math.sqrt(2.0) * math.sqrt(rel0**2 + relt**2)
        noisy = ReadNoise("per_cell", sig, adc_bits=None, dac_bits=None)
        gn = make_generator(p["seed"], f"f00c:noisy:{ck}:{layer}:{t}")
        for trial in range(p["trials"]):
            a_o = gdc_alpha_ours(
                TileReader(lambda: st0, noisy, gn), TileReader(lambda s=stt: s, noisy, gn), n
            )
            ref_out = W0 + s0 * torch.randn(m_, n_, generator=gn, dtype=DTYPE)
            cur_out = Wt + st_ * torch.randn(m_, n_, generator=gn, dtype=DTYPE)
            a_n = float(gdc.apply(cur_out, gdc.readout(ref_out)))
            rd = abs(a_o - a_n) / a_n
            rows.append(
                {
                    "check": "F0.0c",
                    "ckpt_seed": ck,
                    "layer": layer,
                    "t_s": t,
                    "mode": "noisy",
                    "trial": trial,
                    "rel_diff": rd,
                    "noise_bound": bound,
                    "f00c_rel_diff_over_noise_bound": rd / bound,
                }
            )
    return rows


# --------------------------------------------------------------------------- F0.0d


def check_d(p: dict) -> list[dict]:
    from neurofault.config import FaultConfig
    from neurofault.crossbar.health_monitor import HealthMonitor
    from neurofault.faults.injection import inject_faults
    from neurofault.spectral.drift import LegacyRailShift
    from tests.unit.fakes import make_handle

    shape, direction, dist, seed = tuple(p["shape"]), p["direction"], p["distribution"], p["seed"]
    # Dyadic units (r_on=0.25 -> g_max=4, r_off=1 -> g_min=1, magnitude 0.25 -> shift 0.75,
    # base on a 1/64 grid) make every float32 and float64 operation exact, so "bit-identical"
    # is a meaningful, achievable test of selection/sign/clamp logic instead of a rounding test.
    handle = make_handle(shape, r_on=0.25, r_off=1.0)
    mon = HealthMonitor(handle)
    gb = make_generator(seed, "f00d:base")
    base = (1.0 + torch.randint(0, 192, shape, generator=gb).float() / 64.0).float()
    handle.accessor.write(base.clone())
    cfg = FaultConfig(
        density=p["density"],
        failure_model="drift",
        distribution=dist,
        drift_direction=direction,
        drift_magnitude=0.25,
    )
    inject_faults(handle, mon, cfg, generator=make_generator(seed, "f00d:inj"))
    legacy = handle.accessor.read().double()
    enc = EncodingSpec("offset_refcol", g_max_uS=4.0, g_min_frac=0.25)
    st0 = ConductanceState(base.double(), None, 1.0, enc)
    out = (
        LegacyRailShift(p["density"], 0.25, direction, dist)
        .sample(st0, make_generator(seed, "f00d:inj"))
        .at(1.0)
    )
    diff = float((out.g_pos - legacy).abs().max())
    n_changed = int((legacy != base.double()).sum())
    # informational: the same comparison in physical units (S -> uS, float32 legacy) is only
    # float-rounding-equal, which is why the dyadic setting above is the criterion setting
    h2 = make_handle(shape)
    m2 = HealthMonitor(h2)
    b2 = m2.g_min + torch.rand(shape, generator=gb, dtype=torch.float32) * m2.g_range
    h2.accessor.write(b2.clone())
    cfg2 = FaultConfig(
        density=p["density"],
        failure_model="drift",
        distribution=dist,
        drift_direction=direction,
        drift_magnitude=0.25,
    )
    inject_faults(h2, m2, cfg2, generator=make_generator(seed, "f00d:inj2"))
    leg2 = h2.accessor.read().double() * 1e6
    enc2 = EncodingSpec("offset_refcol", g_max_uS=1e6 * m2.g_max, g_min_frac=m2.g_min / m2.g_max)
    s2 = ConductanceState(b2.double() * 1e6, None, 1.0, enc2)
    o2 = (
        LegacyRailShift(p["density"], 0.25, direction, dist)
        .sample(s2, make_generator(seed, "f00d:inj2"))
        .at(1.0)
    )
    return [
        {
            "check": "F0.0d",
            "direction": direction,
            "distribution": dist,
            "n_changed": n_changed,
            "f00d_max_abs_diff": diff,
            "info_physical_units_max_abs_diff_uS": float((o2.g_pos - leg2).abs().max()),
        }
    ]


# --------------------------------------------------------------------------- F0.0e


def check_e(p: dict) -> list[dict]:
    from neurofault.spectral.drift import (
        ComposedDrift,
        LegacyRailShift,
        PCMPerDevice,
        PowerLawUniform,
        RRAMRelaxation,
        StructuredNu,
        ThermalGain,
    )
    from neurofault.spectral.tiles import synthetic_tile

    seed, T = p["seed"], p["T"]
    times = [60.0, 3.6e3, 8.64e4, 3.15e7]
    tile = get_tile(p["ckpt_seed"], p["layer"], 0, 0, T, T, T)
    rows: list[dict] = []

    # --- nu constant across t (and in-place stable), D2 and D3
    for name, model in [
        ("D2", PCMPerDevice()),
        ("D3_rank_r_rho2", StructuredNu("rank_r", 2.0)),
        ("D3_hotspot_rho1", StructuredNu("hotspot", 1.0)),
    ]:
        st0 = encode(tile, EncodingSpec("diff_pair", g_min_frac=0.01))
        real = model.sample(st0, make_generator(seed, f"f00e:nu:{name}"))
        nu_before = real.nu.clone()
        eff_dev = 0.0
        for a, t in enumerate(times):
            st_t = real.at(t)
            if a == 0:
                continue
            ref_t = real.at(times[0])
            free = (~st0.stuck[..., 0]) & (st_t.g_pos > st0.enc.g_min_uS + 1e-9)
            nu_a = -torch.log(st_t.g_pos / st0.g_pos) / math.log((t + T0_S) / T0_S)
            nu_b = -torch.log(ref_t.g_pos / st0.g_pos) / math.log((times[0] + T0_S) / T0_S)
            ok = free & (st0.g_pos > 0) & (ref_t.g_pos > st0.enc.g_min_uS + 1e-9)
            if ok.any():
                eff_dev = max(eff_dev, float((nu_a - nu_b)[ok].abs().max()))
        rows.append(
            {
                "check": "F0.0e",
                "item": f"nu_persistence_{name}",
                "f00e_nu_max_abs_change": float((real.nu - nu_before).abs().max()),
                "info_effective_nu_max_dev": eff_dev,
            }
        )
        # replay determinism of the draw
        real2 = model.sample(st0, make_generator(seed, f"f00e:nu:{name}"))
        rows[-1]["info_resample_nu_max_abs_diff"] = float((real2.nu - nu_before).abs().max())

    # --- D1 + GDC residual (GDC exact for uniform decay when nothing clips)
    for label in ["diff_pair_g0", "signed_ideal", "diff_pair_g0.01", "diff_pair_g0.05"]:
        st0 = encode(tile, enc_from_label(label))
        real = PowerLawUniform().sample(st0, make_generator(seed, "f00e:d1"))
        W0 = st0.weights()
        worst = 0.0
        for t in times:
            Wt = real.at(t).weights()
            alpha = W0.abs().mean() / Wt.abs().mean()
            worst = max(worst, float((alpha * Wt - W0).norm() / W0.norm()))
        key = (
            "f00e_d1_gdc_residual_rel"
            if label in ("diff_pair_g0", "signed_ideal")
            else ("info_d1_gdc_residual_rel_floor_clip")
        )
        rows.append({"check": "F0.0e", "item": f"d1_gdc_{label}", key: worst})

    # --- encode/decode round trip to 1e-12
    worst = 0.0
    tiles = [tile] + [
        get_tile(p["ckpt_seed"], ly, 0, 0, T, T, T) for ly in PROBE_LAYERS if ly != p["layer"]
    ]
    tiles.append(synthetic_tile(64, 48, spectrum="decay", gen=make_generator(seed, "f00e:syn")))
    for t_ in tiles:
        for label in ["signed_ideal", "diff_pair_g0", "diff_pair_g0.01", "diff_pair_g0.05"]:
            worst = max(
                worst, float((encode(t_, enc_from_label(label)).weights() - t_.w).abs().max())
            )
        for gm in (0.0, 0.05):
            enc = EncodingSpec("offset_refcol", g_min_frac=gm)
            worst = max(worst, float((encode(t_, enc).weights() - t_.w).abs().max()))
    rows.append({"check": "F0.0e", "item": "roundtrip", "f00e_roundtrip_max_abs_err": worst})

    # --- stuck devices never move (D7 periphery gain excluded: it sits after the array by design)
    st_clean = encode(tile, EncodingSpec("diff_pair", g_min_frac=0.01))
    models = {
        "D0": LegacyRailShift(0.3),
        "D1": PowerLawUniform(),
        "D2": PCMPerDevice(),
        "D3": StructuredNu("rank_r", 2.0),
        "D4+D2": ThermalGain(PCMPerDevice()),
        "D5_cmo": RRAMRelaxation("cmo"),
        "D5_mean_revert": RRAMRelaxation("mean_revert"),
        "D1+D2_composed": ComposedDrift([PowerLawUniform(), PCMPerDevice()]),
    }
    specs = [
        FaultSpec("stuck_random", k=64, rail="mixed"),
        FaultSpec("stuck_clustered", k=64, rail="mixed"),
        FaultSpec("line_row", k=2),
        FaultSpec("line_col", k=2, line_mode="open"),
    ]
    for mname, model in models.items():
        worst = 0.0
        n_stuck = 0
        for si, spec in enumerate(specs):
            stf, _ = inject(st_clean, spec, make_generator(seed, f"f00e:fault:{si}"))
            real = model.sample(stf, make_generator(seed, f"f00e:stuck:{mname}"))
            for t in times:
                st_t = real.at(t)
                for plane, (g_new, g_old) in enumerate(
                    [(st_t.g_pos, stf.g_pos), (st_t.g_neg, stf.g_neg)]
                ):
                    msk = stf.stuck[..., plane]
                    if msk.any():
                        worst = max(worst, float((g_new[msk] - g_old[msk]).abs().max()))
                        n_stuck += int(msk.sum())
        rows.append(
            {
                "check": "F0.0e",
                "item": f"stuck_{mname}",
                "f00e_stuck_max_abs_move": worst,
                "info_n_stuck_checked": n_stuck,
            }
        )
    return rows


# --------------------------------------------------------------------------- driver


def task(p: dict) -> list[dict]:
    return {"a": check_a, "b": check_b, "c": check_c, "d": check_d, "e": check_e}[p["check_id"]](p)


def build_grid(cfg: dict, seed: int) -> list[dict]:
    g: list[dict] = []
    ck0 = cfg["ckpt_seeds"][0]
    for ck in [ck0]:
        for layer in PROBE_LAYERS:
            g.append(dict(check_id="a", seed=seed, ckpt_seed=ck, layer=layer, T=256,
                          trials=cfg["a_trials"]))  # fmt: skip
    for lv in cfg["b_levels"]:
        g.append(dict(check_id="b", seed=seed, level_frac=lv, n_devices=cfg["b_n_devices"],
                      times_s=cfg["b_times_s"]))  # fmt: skip
    for ck in cfg["ckpt_seeds"]:
        for layer in cfg["c_layers"]:
            g.append(dict(check_id="c", seed=seed, ckpt_seed=ck, layer=layer, T=256,
                          noise_frac=0.02, trials=cfg["c_trials"], times_s=cfg["c_times_s"]))  # fmt: skip
    for direction in ("toward_hrs", "toward_lrs"):
        for dist in ("random", "clustered", "gradient"):
            for s in range(cfg["d_seeds"]):
                g.append(dict(check_id="d", seed=seed + s, direction=direction, distribution=dist,
                              shape=[32, 24], density=0.3))  # fmt: skip
    g.append(dict(check_id="e", seed=seed, ckpt_seed=ck0, layer=PROBE_LAYERS[1], T=cfg["e_T"]))
    return g


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "f0_0.yaml", "f0_0", argv)
    cfg = load_config(a.config, a.smoke)
    cfg["master_seed"] = a.seed
    run_out = __import__("pathlib").Path(a.out)
    grid = build_grid(cfg, a.seed)
    df = run_grid(task, grid, jobs=a.jobs, out_dir=run_out / "_cells_cache")
    show = df[df.check == "F0.0a"][["layer", "sigma1_snr_med", "probe_snr_med", "rowloc_p1",
                                    "gap_new", "gap_orig"]]  # fmt: skip
    print(
        "--- F0.0a on the new stack (reference: sigma1 SNR 0.05-0.09, probe SNR ~3.1, "
        "row-loc 55-62%) ---"
    )
    print(show.to_string(index=False))
    finish(
        "F0.0", "f0_validity.yaml", cfg, a, df,
        extra={"note": "F0.8a-c are out of scope for this script (need bridge); "
                       "overall INCONCLUSIVE is by construction"},
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    raise SystemExit(main())

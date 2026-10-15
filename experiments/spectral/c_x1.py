"""C-X1: settling window and speed of the pilot loop vs the MVM probe on REAL ResNet-9 tiles
(C-X1; criteria/c_claims.yaml S1, K1a-c, D1 inputs, KX1; f0_loop.yaml K6.2).

Cell = (checkpoint seed, layer, tile size N, tile origin); G+/G- (g_min 5 %), dilated two-array
loop with loop-only random-sign flat pilot. Setting = (beta, f, amplitude mode):
beta in {0,.5,1,2,3,5,10}, f in {.1,.25,.5,.75}, amplitude in {clip (hard rail), agc_ideal (the
scratch's instantaneous peak AGC - a CEILING, not a plan arm), limiter_{.01,.03,.1} (peak AGC with
bandwidth x omega), servo_{0.5,2} (G_lambda PI servo, f = 0.5 only: the servo drives G_lambda to
lambda, f only sets its starting point)}.

Per (cell, setting), over `n_faults` single stuck-at-g_max cells (device G+ or G- at random, the
paired state-dependent weight change of faults.inject), the healthy loop is warm-started in its
hardware fixed point (modal vector of D^-1(M - G_lambda), at the amplitude the mode holds), the
faulty array (SAME pilot, SAME G_lambda, its own node loads) is re-settled from that state, and the
paired deflection D(t) = o_faulty(t) - o_healthy(t) is analysed:
  snr_loop        median |D_row(t_end)[i]| / 1 mV   (the scratch's SNR column)
  snr_b1          same for the +-1 probe on the same tile and faults
  sigma_star      output noise at which the +-1 probe localizes the faulty row 90 %
  loc_row/col     argmax|D + sigma_star xi| == truth at the settled end
  t_defl90        noise-free: first tau at which D(t) has 90 % of its settled projection
  t_loc90         first tau at which loc(t) >= 0.9 ABSOLUTE at sigma_star (inf if the loop never
                  gets there). Reading of the plan: K1's text demands "equal localization accuracy
                  (>= 90 % at the same output-referred noise)"; the literal "90 % of the asymptotic
                  accuracy" variant (t_loc90_rel) is degenerate when the asymptote is low (a latched
                  loop has asymptote ~0, so it "reaches 90 % of nothing" at once) and is only reported.
  s1_timing_setting  = t_defl90 <= t_MVM(8b) and latch-free/stationary/purity >= 0.98: the SPEED
                  half of S1, noise-free, used as the filter of K1b/K1c ("the setting achieving S1
                  timing"); K1b then asks what SNR that fast setting has.
  sigma90_loop    noise at which the settled loop localizes 90 %; reads_needed_for_90 =
                  (sigma_star / sigma90_loop)^2 independent noisy reads averaged (K1b's "repeats").
t_MVM(8 b) is mvm_timing on the SAME op-amp model (slowest row; mean-load variant also kept).
Times are in 1/omega; tables also give microseconds at 100 MHz GBW (GBW scaling: x 1/(2 pi GBW)).

Extra tables: gap_variation (eigenvector error of the dilated top pair vs relative gap under
iid device variation, real tiles; the 1/gap issue), variants (A0, G_lambda mismatch, offsets on a
reduced set of settings).

  python experiments/spectral/c_x1.py --jobs 3 --allow-dirty [--smoke]
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from c_common import (
    GBW_UNIT,
    SIGMA_REF,
    T_CMP_S,
    Healthy,
    build,
    draw_faults,
    energy_ratio,
    first_stay,
    flat_pilot,
    localization_curves,
    loop_config,
    make_healthy,
    planes,
    probe_baseline,
    required_gl_precision,
    run,
    smooth3,
    stuck_lrs,
)
from f0_common import CRITERIA_DIR, finish, load_config, parse_args

from neurofault.spectral.circuit import with_config
from neurofault.spectral.harness import run_grid
from neurofault.spectral.types import make_generator
from neurofault.spectral.verdict import evaluate, load_criteria, write_verdict

AMPS_ALL_F = ["clip", "agc_ideal", "limiter_0.01", "limiter_0.03", "limiter_0.1"]
AMPS_F05 = ["servo_0.5", "servo_2"]
CONFIG_NAME = "c_x1.yaml"


def amp_params(amp: str) -> dict:
    """Config overrides of an amplitude-mode label (applied with with_config)."""
    if amp in ("clip", "agc_ideal"):
        return {"amplitude": amp}
    kind, val = amp.split("_", 1)
    if kind == "limiter":
        return {"amplitude": "peak_limiter", "agc_bw_hz": float(val) * GBW_UNIT}
    return {"amplitude": "gl_servo", "servo_kp": float(val)}


def amp_bw_over_gbw(amp: str, lam: float, d_mean: float) -> float:
    """The amplitude controller's bandwidth in units of GBW (K1c's servo_bw_over_gbw):
    limiter = its stated bandwidth; servo = natural frequency kp lambda / (2 D) of the
    critically damped PI loop in tau units (= f_n / GBW); clip has no controller (0);
    the instantaneous AGC has infinite bandwidth."""
    if amp == "clip":
        return 0.0
    if amp == "agc_ideal":
        return math.inf
    kind, val = amp.split("_", 1)
    if kind == "limiter":
        return float(val)
    return float(val) * lam / (2.0 * d_mean)


def start_amplitude(amp: str, r1: float, v_sat: float, vt: float) -> float:
    """Peak the healthy loop holds: the target, except the finite-gain limiter which sits at
    vt (1 + r1/kappa) (capped just below the rail, where it latches)."""
    if amp.startswith("limiter_"):
        kappa = float(amp.split("_", 1)[1])
        return min(vt * (1.0 + r1 / kappa), 0.95 * v_sat)
    return vt


def noise_for_90(final_u: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> float:
    """Output noise std at which the settled deflection localizes the faulty row 90 %."""
    f, k = final_u.shape
    xi = rng.standard_normal((f, 48, k))
    a = np.abs(final_u[:, None, :])
    del a
    lo, hi = 1e-6, 1.0
    arg = np.argmax(np.abs(final_u[:, None, :] + lo * xi), axis=-1)
    if float(np.mean(arg == rows[:, None])) < 0.9:
        return math.nan  # even noise-free the settled argmax misses the row: no averaging helps
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        arg = np.argmax(np.abs(final_u[:, None, :] + mid * xi), axis=-1)
        if float(np.mean(arg == rows[:, None])) >= 0.9:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def analyse_setting(
    ctx: dict,
    h: Healthy,
    mh,
    faulty: list,
    vec0: np.ndarray,
    rates: np.ndarray,
    gl_prec: float,
    amp: str,
    probe,
    cell_rng: np.random.Generator,
    seed: int,
) -> dict:
    m = h.gp.shape[0]
    cfg0 = mh.cfg
    over = amp_params(amp)
    model_h = with_config(mh, **over)
    vt = cfg0.v_target_frac * cfg0.v_sat
    r1 = float(rates[0])
    r_rel = float(rates[0] - rates[1])
    d_mean = float(mh.topo.load.mean() + mh.g_nominal)
    bw_ratio = amp_bw_over_gbw(amp, mh.ref_value, d_mean)
    amp0 = start_amplitude(amp, r1, cfg0.v_sat, vt)
    v_start = vec0 * (amp0 / np.abs(vec0).max())
    t_end = 8.0 / r_rel
    if amp.startswith("servo_"):
        t_end = max(t_end, 6.0 / max(bw_ratio, 1e-6))
    t_end = min(t_end, ctx["t_end_cap"])
    dt = min(model_h.default_dt_tau(), t_end / 150.0)
    ns = ctx["n_samples"]
    tr_h = run(model_h, v_start, t_end, dt, ns, seed)
    oh = tr_h.v.numpy()
    t = tr_h.t.numpy()
    d_u, d_v, rows, cols = [], [], [], []
    latched_f = 0
    for mf, (i, j, _dev) in faulty:
        tr_f = run(with_config(mf, **over), v_start, t_end, dt, ns, seed)
        d = tr_f.v.numpy() - oh
        d_u.append(d[:, :m])
        d_v.append(d[:, m:])
        rows.append(i)
        cols.append(j)
        latched_f += int(tr_f.latched)
    d_u, d_v = np.stack(d_u), np.stack(d_v)
    rows, cols = np.array(rows), np.array(cols)
    nf = len(rows)
    # --- noise-free metrics
    final_u = d_u[:, -1, :]
    snr_loop = float(np.median(np.abs(final_u[np.arange(nf), rows])) / SIGMA_REF)
    proj = (
        np.einsum("ftk,fk->ft", d_u, final_u)
        / np.maximum(np.einsum("fk,fk->f", final_u, final_u), 1e-300)[:, None]
    )
    t_defl = [first_stay(t, p >= 0.9) for p in proj]
    t_defl90 = float(np.median(t_defl))
    # --- localization at the probe's 90 % noise
    loc_r, loc_c = localization_curves(
        d_u, d_v, rows, cols, probe.sigma_star, cell_rng, ctx["noise_draws"]
    )
    lr = smooth3(loc_r)
    loc_inf = float(np.mean(loc_r[-3:]))
    loc_inf_c = float(np.mean(loc_c[-3:]))
    t_loc90_rel = first_stay(t, lr >= 0.9 * loc_inf) if loc_inf > 0 else math.inf
    t_loc90 = first_stay(t, lr >= 0.9)  # absolute: equal accuracy at the probe's noise (K1 text)
    sigma90 = noise_for_90(final_u, rows, cell_rng)
    # --- state quality
    purity = float(tr_h.purity_t[-1])
    latched = bool(tr_h.latched or latched_f > nf // 2)
    peak_end = float(tr_h.peak[-1])
    peak_drift = peak_end / max(float(tr_h.peak[0]), 1e-300)
    stationary = 0.8 <= peak_drift <= 1.25  # unregulated (clip) modes grow or shrink
    amp_ok = (not latched) and stationary and purity >= 0.98
    t_mvm = probe.t_mvm8
    timing_ok = bool(t_defl90 <= 1.0 * t_mvm and amp_ok)
    gbw100 = 1e8
    t_cmp_tau = T_CMP_S * 2 * math.pi * gbw100
    n_amp_arrays = 2  # dilation: A and its transposed copy
    row = {
        "t_end": t_end,
        "snr_loop": snr_loop,
        "snr_loop_over_snr_b1": snr_loop / probe.snr_ref,
        "sigma90_loop": sigma90,
        "reads_needed_for_90": (probe.sigma_star / sigma90) ** 2
        if not math.isnan(sigma90)
        else math.inf,
        "loc_row_noisefree": float(np.mean(np.argmax(np.abs(final_u), axis=1) == rows)),
        "loc_row_inf": loc_inf,
        "loc_col_inf": loc_inf_c,
        "t_defl90": t_defl90,
        "t_loc90": t_loc90,
        "t_loc90_rel": t_loc90_rel,
        "t_loc90_over_mvm8": t_loc90 / t_mvm,
        "t_loc90_gt_3x_mvm8": float(t_loc90 > 3.0 * t_mvm),
        "t_defl90_gt_3x_mvm8": float(t_defl90 > 3.0 * t_mvm),
        "t_defl90_over_mvm8": t_defl90 / t_mvm,
        "t_defl90_over_mvm8_mean": t_defl90 / probe.t_mvm8_mean,
        "s1_timing_setting": float(timing_ok),
        "s1_cell_ok": float(t_loc90 <= 1.0 * t_mvm and loc_inf >= 0.9 and amp_ok),
        "latched": float(latched),
        "stationary": float(stationary),
        "purity_end": purity,
        "peak_end_over_vsat": peak_end / cfg0.v_sat,
        "required_gl_precision_frac": gl_prec,
        "servo_bw_over_gbw": bw_ratio,
        "warm_latency_over_t_mvm_latch_free": (t_defl90 / t_mvm) if amp_ok else np.nan,
        "t_loop_over_t_mvm_cmp": t_defl90 / (t_mvm + t_cmp_tau),
        "energy_loop_over_mvm_cmp": energy_ratio(
            h.gp.shape[0], n_amp_arrays, t_defl90, t_mvm + t_cmp_tau, gbw100
        ),
        "matched_detection": bool(loc_inf >= 0.9),
        "t_loop_over_t_mvm": (t_loc90 / t_mvm) if loc_inf >= 0.9 else np.nan,
        "t_defl90_us_100M": t_defl90 / (2 * math.pi * gbw100) * 1e6,
        "t_mvm8_us_100M": t_mvm / (2 * math.pi * gbw100) * 1e6,
        "is_ideal_ref": float(amp == "agc_ideal"),
    }
    return row


def stuck_models(h: Healthy, cfg, faults, seed: int):
    out = []
    for i, j, dev in faults:
        gpf, gnf, _ = stuck_lrs(h.gp, h.gn, i, j, dev)
        out.append((build(h, cfg, seed, gpf, gnf), (i, j, dev)))
    return out


def gap_variation_rows(point: dict, gp, gn, u0, v0, seed: int) -> list[dict]:
    """Eigenvector (dilation top pair) error under iid multiplicative device variation, on the
    real tile, vs the relative gap 1 - sigma_2'/sigma_1' that the pilot sets. The fixed point of
    the servo-settled loop equals this eigenvector (V1/V6p), so no ODE is needed."""
    rows = []
    n_draws = point["gv_draws"]
    rng = np.random.default_rng(seed)
    a = gp - gn
    s1_a = float(np.linalg.norm(a, 2))
    for beta in point["gv_betas"]:
        p = beta * s1_a * np.outer(u0, v0)
        u, s, vh = np.linalg.svd(a + p)
        g_rel = 1.0 - float(s[1] / s[0])
        vec0 = np.concatenate([u[:, 0], vh[0]])
        for sig in point["gv_sigmas"]:
            errs = []
            for _ in range(n_draws):
                gp2 = gp * (1.0 + sig * rng.standard_normal(gp.shape))
                gn2 = gn * (1.0 + sig * rng.standard_normal(gn.shape))
                u2, _, vh2 = np.linalg.svd(np.clip(gp2, 0, None) - np.clip(gn2, 0, None) + p)
                v2 = np.concatenate([u2[:, 0], vh2[0]])
                v2 = v2 if v2 @ vec0 > 0 else -v2
                errs.append(float(np.linalg.norm(v2 - vec0)))
            rows.append(
                {
                    "kind": "gap_variation",
                    "beta": beta,
                    "sigma_var": sig,
                    "g_rel": g_rel,
                    "gap_ratio": 1.0 / (1.0 - g_rel),
                    "eigvec_err_median": float(np.median(errs)),
                    "eigvec_err_p90": float(np.quantile(errs, 0.9)),
                }
            )
    return rows


def cell_task(point: dict) -> list[dict]:
    seed, layer, n = point["seed"], point["layer"], point["N"]
    r0, c0 = point["r0"], point["c0"]
    t0 = time.time()
    gp, gn = planes(seed, layer, r0, c0, n)
    gen = make_generator(seed, f"c_x1:pilot:{layer}:{n}:{r0}:{c0}")
    u0, v0 = flat_pilot(n, n, gen)
    cell_rng = np.random.default_rng([seed, n, r0, c0, len(layer)])
    faults_all = draw_faults(n, n, point["n_cal_faults"], cell_rng)
    probe = probe_baseline(gp, gn, faults_all, cell_rng)
    faults = faults_all[: point["n_faults"]]
    sv_a = np.linalg.svd(gp - gn, compute_uv=False)
    out: list[dict] = []
    base = {
        "kind": "setting",
        "variant": "base",
        "gap_a": float(sv_a[0] / sv_a[1]),
        "t_mvm8": probe.t_mvm8,
        "t_mvm8_mean": probe.t_mvm8_mean,
        "sigma_star": probe.sigma_star,
        "snr_b1": probe.snr_ref,
        "loc_b1_at_1mv": probe.loc_at_ref,
    }
    ctx = {
        "t_end_cap": point["t_end_cap"],
        "n_samples": point["n_samples"],
        "noise_draws": point["noise_draws"],
    }
    variants = [{"name": "base"}] + (point["variants"] if point["variants"] else [])
    for var in variants:
        vname = var["name"]
        sub_settings = point["settings"] if vname == "base" else var["settings"]
        cfg_kw = {k: v for k, v in var.items() if k not in ("name", "settings")}
        if vname == "base":
            cfg_kw.update(point.get("base_overrides") or {})
        by_bf: dict = {}
        for beta, f, amp in sub_settings:
            by_bf.setdefault((beta, f), []).append(amp)
        for (beta, f), amps in by_bf.items():
            h = make_healthy(gp, gn, float(beta), float(f), u0, v0)
            cfg = loop_config(h, "clip", **cfg_kw)
            mh = build(h, cfg, seed)
            rates, vecs = mh.modal_rates(2, vectors=True)
            vec0 = vecs[:, 0] if vecs[np.argmax(np.abs(vecs[:, 0])), 0] > 0 else -vecs[:, 0]
            gl_prec = required_gl_precision(mh, cell_rng)
            fmods = stuck_models(h, cfg, faults, seed)
            for amp in amps:
                row = analyse_setting(
                    ctx, h, mh, fmods, vec0, rates, gl_prec, amp, probe, cell_rng, seed
                )
                row.update(base)
                row.update(
                    {
                        "variant": vname,
                        "beta": float(beta),
                        "f": float(f),
                        "amp_mode": amp,
                        "gap": h.gap,
                        "delta": h.delta,
                        "s_over_s1": float(mh.topo.load.mean() / h.sig[0]),
                        "r1": float(rates[0]),
                        "r_rel": float(rates[0] - rates[1]),
                    }
                )
                out.append(row)
    if point["gv_betas"]:
        out += gap_variation_rows(point, gp, gn, u0, v0, seed)
    print(
        f"cell {seed}/{layer}/N{n}/({r0},{c0}) done in {time.time() - t0:.0f}s: {len(out)} rows",
        flush=True,
    )
    return out


def build_grid(cfg: dict) -> list[dict]:
    betas, fs = cfg["betas"], cfg["fs"]
    settings = [(b, f, a) for b in betas for f in fs for a in cfg.get("amps_all_f", AMPS_ALL_F)]
    settings += [(b, 0.5, a) for b in betas for a in cfg.get("amps_f05", AMPS_F05)]
    pts = []
    for seed in cfg["ckpt_seeds"]:
        for n, layers in cfg["tiles"].items():
            for layer in layers:
                pts.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "N": int(n),
                        "r0": 0,
                        "c0": 0,
                        "settings": settings,
                        "variants": (
                            cfg["variants"]
                            if (seed == cfg["ckpt_seeds"][0] and layer == layers[0])
                            else None
                        ),
                        "base_overrides": cfg.get("base_overrides"),
                        "n_faults": cfg["n_faults"],
                        "n_cal_faults": cfg["n_cal_faults"],
                        "t_end_cap": cfg["t_end_cap"],
                        "n_samples": cfg["n_samples"],
                        "noise_draws": cfg["noise_draws"],
                        "gv_betas": cfg["gv_betas"],
                        "gv_sigmas": cfg["gv_sigmas"],
                        "gv_draws": cfg["gv_draws"],
                    }
                )
    return pts


def summarize(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = m[(m["kind"] == "setting") & (m["variant"] == "base")].copy()
    g = (
        base.groupby(["N", "beta", "f", "amp_mode"])
        .agg(
            gap=("gap", "median"),
            snr_ratio=("snr_loop_over_snr_b1", "median"),
            loc_row=("loc_row_inf", "median"),
            loc_col=("loc_col_inf", "median"),
            t_defl90=("t_defl90", "median"),
            t_loc90=("t_loc90", "median"),
            t_mvm8=("t_mvm8", "median"),
            ratio_loc90=("t_loc90_over_mvm8", "median"),
            frac_t3=("t_loc90_gt_3x_mvm8", "mean"),
            frac_timing=("s1_timing_setting", "mean"),
            frac_s1=("s1_cell_ok", "mean"),
            latched=("latched", "mean"),
            purity=("purity_end", "median"),
            cells=("seed", "count"),
        )
        .reset_index()
    )
    return {"summary_by_setting": g}


def main(argv=None) -> None:
    args = parse_args("C-X1 loop vs MVM probe on real tiles", CONFIG_NAME, "c_x1", argv)
    cfg = load_config(args.config, args.smoke)
    points = build_grid(cfg)
    t0 = time.time()
    out_dir = Path(args.out) / "_cells_cache"
    metrics = run_grid(cell_task, points, jobs=args.jobs, out_dir=out_dir)
    print(f"grid done in {time.time() - t0:.0f}s, {len(metrics)} rows", flush=True)
    side = summarize(metrics)
    gv = metrics[metrics["kind"] == "gap_variation"]
    if len(gv):
        side["gap_variation"] = gv.dropna(axis=1, how="all")
    metrics_c = metrics[metrics["kind"] != "gap_variation"].copy()
    for c in ("matched_detection",):
        metrics_c[c] = metrics_c[c].astype(float)
    # criteria are judged on the plan's arms: base variant, ideal-AGC ceiling excluded
    judged = metrics_c[(metrics_c["variant"] == "base") & (metrics_c["is_ideal_ref"] == 0)]
    run_dir, _ = finish("C-X1 (plan arms)", "c_claims.yaml", cfg, args, judged, side_tables=side)
    metrics.to_csv(run_dir / "metrics_all_rows.csv", index=False)
    ceil = metrics_c[(metrics_c["variant"] == "base")]
    v = evaluate(load_criteria(CRITERIA_DIR / "c_claims.yaml"), ceil)
    write_verdict(v, run_dir / "verdict_incl_ideal_agc_ceiling.json")
    print("verdict incl. ideal-AGC ceiling:", {c["id"]: c["status"] for c in v["criteria"]})


if __name__ == "__main__":
    main()

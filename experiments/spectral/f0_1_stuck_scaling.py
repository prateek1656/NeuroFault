# ruff: noqa: C408
"""F0.1 rigorous re-run of the sparse-fault feasibility probe (F0.1, H1.1-H1.6).

Question: can the dominant singular value sigma_1 of a real trained crossbar tile see stuck
cells, and how does its shift scale with tile size and fault count?  Compared against a +-1 probe
residual (B4) and a 1-row checksum (B4b) on the same noisy reads.

Per (ckpt seed, layer, tile size, tile, encoding) task, with fresh read noise every readout:
  * negatives: n_neg healthy noisy readouts; positives: n_pos faulty noisy readouts per fault cfg.
  * sigma_1 score = |sigma_1(noisy full readout) - sigma_1(noise-free healthy) - mean_neg(...)|
    (calibrated against the same noise, so the noise-induced bias of sigma_1 does not leak into
    the detection score). Exact SVD of the readout (the old probe's simplification, now tested).
  * probe stat = max_i |r_i| (P=1) / max_i sum_p r_ip^2 (P=8) with r = noisy W x - stored W x.
  * checksum stat = |1^T r| (one MVM).  Cost ratio is counted in MVM-equivalents: checksum 1 vs
    min(m, n) for reading the tile out for a digital SVD.
  * noise-free relative shift |sigma_1(faulty) - sigma_1| / sigma_1 for the scaling fits.
Tile-level rows are reduced to the plan's cells (layer, tile size, encoding, seed) by the median
over tiles before verdict.py sees them. `k` is the number of faulty CELLS (a line fault has k =
tile width/height), so H1.1's k = 1 filter selects only single-cell faults.

Not done (stated, not hidden): fault amplitude 0.5 (faults.inject only pins devices at rails) and
the top-r singular-vector deformation signature (not referenced by any frozen criterion).

  python experiments/spectral/f0_1_stuck_scaling.py --jobs 8 --allow-dirty
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from f0_common import (
    PROBE_LAYERS,
    FaultSpec,
    M,
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
    sigma1,
)

from neurofault.spectral.harness import run_grid
from neurofault.spectral.noise import program
from neurofault.spectral.types import DTYPE, ProgNoise

METRIC_COLS = [
    "sigma1_auc", "sigma1_det_fpr1", "sigma1_snr", "rel_shift_median", "probe_p1_auc",
    "probe_p8_auc", "checksum_auc", "rowloc_p1", "rowloc_p8", "colloc_p8", "cellloc_p8",
    "dw_over_wmax_mean", "dw_invisible_frac", "checksum_minus_spectral_auc",
    "checksum_to_spectral_cost_ratio",
]  # fmt: skip
ID_COLS = [
    "setting", "stage", "ckpt_seed", "layer", "tile_size", "encoding", "noise_mode",
    "noise_sigma_frac", "prog_noise", "fault_kind", "k", "n_lines", "rail",
]  # fmt: skip


# --------------------------------------------------------------------------- catalog


def fault_catalog(k_sweep, clustered_k, n_pos, n_shift, lines=True, open_lines=True):
    """List of fault configs: stuck_random k-sweep (rail mixed), clustered, lines, and the
    noise-free lrs shift sweep used for the H1.2 fits."""
    cat = []
    for k in k_sweep:
        cat.append(dict(fid=f"rand_k{k}", fault_kind="stuck_random", spec_kind="stuck_random",
                        k=k, rail="mixed", line_mode="stuck_rail", n=n_pos))  # fmt: skip
    for k in clustered_k:
        cat.append(dict(fid=f"clus_k{k}", fault_kind="stuck_clustered", spec_kind="stuck_clustered",
                        k=k, rail="mixed", line_mode="stuck_rail", n=n_pos))  # fmt: skip
    if lines:
        for axis in ("row", "col"):
            cat.append(dict(fid=f"line_{axis}", fault_kind=f"line_{axis}", spec_kind=f"line_{axis}",
                            k=1, rail="mixed", line_mode="stuck_rail", n=n_pos))  # fmt: skip
            if open_lines:
                cat.append(dict(fid=f"line_{axis}_open", fault_kind=f"line_{axis}_open",
                                spec_kind=f"line_{axis}", k=1, rail="mixed", line_mode="open",
                                n=n_pos))  # fmt: skip
    return cat


def shift_catalog(k_sweep, n_shift):
    return [
        dict(fid=f"lrs_k{k}", fault_kind="stuck_random_lrs", spec_kind="stuck_random", k=k,
             rail="lrs", line_mode="stuck_rail", n=n_shift, shift_only=True)
        for k in k_sweep
    ]  # fmt: skip


def make_noise(c: dict) -> ReadNoise:
    s, mode = c["sigma"], c["mode"]
    return ReadNoise(
        mode,
        sigma_cell_frac=s if mode in ("per_cell", "both") else 0.0,
        sigma_out_frac=s if mode in ("output_referred", "both") else 0.0,
        adc_bits=None,
        dac_bits=None,
    )


# --------------------------------------------------------------------------- worker


def _probe_stats(reader, W_h, X8, bwd_Y=None):
    R = reader.mvm(X8) - W_h @ X8
    sc1, sc8 = R[:, 0].abs(), (R**2).sum(1)
    out = {
        "p1": float(sc1.max()), "p8": float(sc8.max()), "ck": float(R[:, 0].sum().abs()),
        "sc1": sc1, "sc8": sc8,
    }  # fmt: skip
    if bwd_Y is not None:
        Rc = reader.mvm_t(bwd_Y) - W_h.T @ bwd_Y
        out["col8"] = (Rc**2).sum(1)
    return out


def _signs(shape, gen):
    x = torch.sign(torch.randn(*shape, generator=gen, dtype=DTYPE))
    x[x == 0] = 1.0
    return x


def task_old_probe(p: dict) -> list[dict]:
    r = run_old_probe(p["ckpt_seed"], p["layer"], p["master_seed"], T=256, trials=p["trials"],
                      n_noise_draws=20, p_multi=8)  # fmt: skip
    return [{
        "row_type": "old_probe", "setting": "old_probe", "ckpt_seed": p["ckpt_seed"],
        "layer": p["layer"], "tile_size": 256, "encoding": "signed_ideal", "noise_mode": "per_cell",
        "noise_sigma_frac": 0.02, "k": 1, "fault_kind": "stuck_random", "rowloc_p1": r["rowloc_p1"],
        "rowloc_p8": r["rowloc_p8"], "sigma1_snr": r["sigma1_snr_med"],
        "probe_snr_med": r["probe_snr_med"],
    }]  # fmt: skip


def task_tile(p: dict) -> list[dict]:
    ms = p["master_seed"]
    tile = get_tile(p["ckpt_seed"], p["layer"], p["r0"], p["c0"], p["m"], p["n"], p["tile_size"])
    enc_label = p["encoding"]
    st0 = encode(tile, enc_from_label(enc_label))
    W0 = st0.weights()
    m, n = W0.shape
    wmax = float(W0.abs().max())
    s1_h = sigma1(W0)
    tkey = f"{p['ckpt_seed']}:{p['layer']}:{p['tile_size']}:{p['tile_idx']}"
    conds = p["conds"]
    base = {
        "row_type": "tile", "setting": "grid", "stage": p["stage"], "ckpt_seed": p["ckpt_seed"],
        "layer": p["layer"], "tile_size": p["tile_size"], "tile_idx": p["tile_idx"],
        "tile_m": m, "tile_n": n, "encoding": enc_label,
    }  # fmt: skip
    n_neg = p["n_neg"]

    # ---- negatives (healthy, fresh noise) per condition
    neg: dict[str, dict] = {}
    for c in conds:
        noise = make_noise(c)
        gn = make_generator(ms, f"f01:{tkey}:{enc_label}:{c['id']}:noise")
        prog = None if c["prog"] == "none" else ProgNoise(c["prog"])
        d, p1, p8, ck = [], [], [], []
        for i in range(n_neg):
            if prog is None:
                st_h, s1h = st0, s1_h
            else:
                st_h = program(
                    st0, prog, make_generator(ms, f"f01:{tkey}:{enc_label}:{c['id']}:pn{i}")
                )
                s1h = sigma1(st_h.weights())
            rd = TileReader(lambda s=st_h: s, noise, gn)
            d.append(sigma1(rd.full_readout()) - s1h)
            X8 = _signs((n, 8), make_generator(ms, f"f01:{tkey}:{enc_label}:{c['id']}:nx{i}"))
            ps = _probe_stats(rd, st_h.weights(), X8)
            p1.append(ps["p1"]), p8.append(ps["p8"]), ck.append(ps["ck"])
        d = np.array(d)
        neg[c["id"]] = dict(d=d, mean=float(d.mean()), std=float(d.std()), p1=p1, p8=p8, ck=ck)

    # ---- positives
    acc: dict[tuple, dict] = {}
    shift_acc: dict[str, dict] = {}
    for fs in p["faults"]:
        spec = FaultSpec(fs["spec_kind"], k=fs["k"], rail=fs["rail"], line_mode=fs["line_mode"])
        conds_f = [] if fs.get("shift_only") else [c for c in conds if fs["fid"] in c["fault_ids"]]
        sh = shift_acc.setdefault(fs["fid"], dict(rel=[], abs=[], dw=[], inv=[], ncell=0))
        for t in range(fs["n"]):
            gname = f"f01:{tkey}:{fs['fid']}:{t}"
            st_f0, truth = inject(st0, spec, make_generator(ms, gname))
            sh["ncell"] = len(truth.cells)
            s1_f = sigma1(st_f0.weights())
            sh["abs"].append(abs(s1_f - s1_h))
            sh["rel"].append(abs(s1_f - s1_h) / s1_h)
            dwa = truth.delta_w.abs() / wmax
            sh["dw"].append(float(dwa.mean()))
            sh["inv"].append(float((dwa < 1e-9).double().mean()))
            for c in conds_f:
                noise = make_noise(c)
                a = acc.setdefault(
                    (c["id"], fs["fid"]),
                    dict(d=[], p1=[], p8=[], ck=[], r1=[], r8=[], c8=[], cell=[], ncell=0),
                )
                if "gn" not in a:
                    a["gn"] = make_generator(
                        ms, f"f01:{tkey}:{enc_label}:{c['id']}:noisep:{fs['fid']}"
                    )
                gn = a["gn"]
                if c["prog"] == "none":
                    st_h, st_f, s1h, tr = st0, st_f0, s1_h, truth
                else:
                    st_h = program(
                        st0, ProgNoise(c["prog"]),
                        make_generator(ms, f"f01:{tkey}:{enc_label}:{c['id']}:pp:{fs['fid']}:{t}"),
                    )  # fmt: skip
                    st_f, tr = inject(st_h, spec, make_generator(ms, gname))
                    s1h = sigma1(st_h.weights())
                rd = TileReader(lambda s=st_f: s, noise, gn)
                a["d"].append(sigma1(rd.full_readout()) - s1h)
                gpn = make_generator(ms, f"f01:{tkey}:{enc_label}:{c['id']}:{fs['fid']}:{t}:probe")
                X8, Y8 = _signs((n, 8), gpn), _signs((m, 8), gpn)
                ps = _probe_stats(rd, st_h.weights(), X8, Y8)
                a["p1"].append(ps["p1"]), a["p8"].append(ps["p8"]), a["ck"].append(ps["ck"])
                nr, nc = len(tr.rows), len(tr.cols)
                rr = rc = None
                if nr < m:
                    r1 = ps["sc1"].topk(nr).indices
                    r8 = ps["sc8"].topk(nr).indices
                    a["r1"].append(M.row_accuracy(r1, tr)), a["r8"].append(M.row_accuracy(r8, tr))
                    rr = a["r8"][-1]
                if nc < n:
                    a["c8"].append(M.col_accuracy(ps["col8"].topk(nc).indices, tr))
                    rc = a["c8"][-1]
                if len(tr.cells) == 1 and rr is not None and rc is not None:
                    a["cell"].append(float(rr == 1.0 and rc == 1.0))
                a["ncell"] = len(tr.cells)

    rows: list[dict] = []
    for fs in p["faults"]:
        sh = shift_acc[fs["fid"]]
        common = {
            **base, "fault_kind": fs["fault_kind"], "k": sh["ncell"],
            "n_lines": fs["k"] if fs["spec_kind"].startswith("line") else 0, "rail": fs["rail"],
            "rel_shift_median": float(np.median(sh["rel"])),
            "dw_over_wmax_mean": float(np.mean(sh["dw"])),
            "dw_invisible_frac": float(np.mean(sh["inv"])),
        }  # fmt: skip
        if fs.get("shift_only"):
            rows.append({**common, "noise_mode": "none", "noise_sigma_frac": 0.0,
                         "prog_noise": "none"})  # fmt: skip
            continue
        for c in conds:
            if fs["fid"] not in c["fault_ids"]:
                continue
            a, ng = acc[(c["id"], fs["fid"])], neg[c["id"]]
            mu = ng["mean"]
            score = lambda v, mu=mu: np.abs(np.asarray(v) - mu)
            pos, nn = score(a["d"]), score(ng["d"])
            s1_auc = M.auc(pos, nn)
            ck_auc = M.auc(a["ck"], ng["ck"])
            row = {
                **common, "noise_mode": c["mode"], "noise_sigma_frac": c["sigma"],
                "prog_noise": c["prog"],
                "sigma1_auc": s1_auc,
                "sigma1_det_fpr1": M.detection_at_fpr(pos, nn, 0.01),
                "probe_p1_auc": M.auc(a["p1"], ng["p1"]),
                "probe_p8_auc": M.auc(a["p8"], ng["p8"]),
                "checksum_auc": ck_auc,
                "checksum_minus_spectral_auc": ck_auc - s1_auc,
                "checksum_to_spectral_cost_ratio": 1.0 / min(m, n),
                "rowloc_p1": float(np.mean(a["r1"])) if a["r1"] else np.nan,
                "rowloc_p8": float(np.mean(a["r8"])) if a["r8"] else np.nan,
                "colloc_p8": float(np.mean(a["c8"])) if a["c8"] else np.nan,
                "cellloc_p8": float(np.mean(a["cell"])) if a["cell"] else np.nan,
            }  # fmt: skip
            if c["prog"] == "none" and ng["std"] > 0:
                row["sigma1_snr"] = float(np.median(sh["abs"]) / ng["std"])
            rows.append(row)
    return rows


def task(p: dict) -> list[dict]:
    return task_old_probe(p) if p["kind"] == "old" else task_tile(p)


# --------------------------------------------------------------------------- grid


def build_grid(cfg: dict) -> list[dict]:
    ms = cfg["master_seed"]
    ksw = cfg["k_sweep"]
    head_faults = fault_catalog(ksw, cfg["clustered_k"], cfg["n_pos"], cfg["n_shift"])
    shift_f = shift_catalog(ksw, cfg["n_shift"])
    head_conds = [dict(id="pc2", mode="per_cell", sigma=0.02, prog="none",
                       fault_ids=[f["fid"] for f in head_faults])]  # fmt: skip
    ofat_fids = [f"rand_k{k}" for k in cfg["ofat_k"]] + ["line_row", "line_col"]
    ofat_faults = [f for f in fault_catalog(ksw, [], cfg["n_pos"], cfg["n_shift"], open_lines=False)
                   if f["fid"] in ofat_fids]  # fmt: skip
    ofat_conds = []
    for mode, s in cfg["ofat_noise"]:
        ofat_conds.append(
            dict(id=f"{mode}{s}", mode=mode, sigma=s, prog="none", fault_ids=ofat_fids)
        )
    if cfg.get("ofat_prog"):
        ofat_conds.append(dict(id="pc2_prog", mode="per_cell", sigma=0.02, prog="aihwkit_pcm_poly",
                               fault_ids=ofat_fids))  # fmt: skip
    grid: list[dict] = []
    for ck in cfg["ckpt_seeds"]:
        mats = {ly: ckpt_mats(ck)[ly].shape for ly in cfg["layers"]}
        for T in cfg["tile_sizes"]:
            for ly in cfg["layers"]:
                M_, N_ = mats[ly]
                origins = [
                    (r0, c0, min(T, M_ - r0), min(T, N_ - c0))
                    for r0 in range(0, M_, T)
                    for c0 in range(0, N_, T)
                    if min(T, M_ - r0) == min(T, M_) and min(T, N_ - c0) == min(T, N_)
                ]
                g = make_generator(ms, f"f01:tiles:{ck}:{ly}:{T}")
                order = torch.randperm(len(origins), generator=g).tolist()
                for ti, oi in enumerate(order[: cfg["max_tiles"]]):
                    r0, c0, m, n = origins[oi]
                    tp = dict(kind="tile", master_seed=ms, ckpt_seed=ck, layer=ly, tile_size=T,
                              tile_idx=ti, r0=r0, c0=c0, m=m, n=n, n_neg=cfg["n_neg"])  # fmt: skip
                    encs = list(cfg["enc_full"]) + (
                        list(cfg["enc_light"]) if ti < cfg["max_tiles_light"] else []
                    )
                    for e in encs:
                        grid.append({**tp, "stage": "headline", "encoding": e, "conds": head_conds,
                                     "faults": head_faults + shift_f})  # fmt: skip
                    if ti == 0:
                        for e in cfg["enc_ofat"]:
                            grid.append({**tp, "stage": "ofat", "encoding": e, "conds": ofat_conds,
                                         "faults": ofat_faults, "n_neg": cfg["n_neg"]})  # fmt: skip
        for ly in PROBE_LAYERS:
            grid.append(dict(kind="old", master_seed=ms, ckpt_seed=ck, layer=ly,
                             trials=cfg["old_probe_trials"]))  # fmt: skip
    return grid


# --------------------------------------------------------------------------- aggregation


def _loglog_slope(x, y) -> float:
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def aggregate(tile_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """tile rows -> cell rows (median over tiles) + old_probe rows + H1.2 fits + H1.4 summary."""
    info: dict = {}
    grid_rows = tile_df[tile_df.row_type == "tile"].copy()
    old = tile_df[tile_df.row_type == "old_probe"].copy()
    ids = [c for c in ID_COLS if c in grid_rows.columns]
    mcols = [c for c in METRIC_COLS if c in grid_rows.columns]
    gb = grid_rows.groupby(ids, dropna=False)
    cells = gb[mcols].median().reset_index()
    cells["n_tiles"] = gb.size().values
    cells["row_type"] = "cell"

    out = [cells, old]
    # ---- H1.2 fits (noise-free relative shift, lrs random-sign incoherent faults)
    sh = grid_rows[grid_rows.fault_kind == "stuck_random_lrs"]
    fits = []
    for (ck, ly, enc), g in sh[sh.k == 1].groupby(["ckpt_seed", "layer", "encoding"]):
        sq = g[(g.tile_m == g.tile_size) & (g.tile_n == g.tile_size)]
        per_size = sq.groupby("tile_size").rel_shift_median.median()
        geo = g.assign(geo=np.sqrt(g.tile_m * g.tile_n)).groupby("tile_size").agg(
            geo=("geo", "median"), s=("rel_shift_median", "median")
        )  # fmt: skip
        row = {"row_type": "h12", "ckpt_seed": ck, "layer": ly, "encoding": enc}
        if len(per_size) >= 3:
            row["h12_slope_tile"] = _loglog_slope(per_size.index, per_size.values)
        if len(geo) >= 3:
            row["info_h12_slope_tile_geom_all_shapes"] = _loglog_slope(geo.geo, geo.s)
        if len(geo) >= 3:
            row["info_h12_slope_tile_nominal_all_shapes"] = _loglog_slope(geo.index, geo.s)
        fits.append(row)
    for (ck, ly, T, enc), g in sh.groupby(["ckpt_seed", "layer", "tile_size", "encoding"]):
        per_k = g.groupby("k").rel_shift_median.median()
        if len(per_k) >= 3:
            fits.append({"row_type": "h12", "ckpt_seed": ck, "layer": ly, "tile_size": T,
                         "encoding": enc, "h12_slope_k": _loglog_slope(per_k.index, per_k.values)})  # fmt: skip
    # informational: mixed-rail (zeroing + saturating, partly coherent) k-slope
    mx = grid_rows[(grid_rows.fault_kind == "stuck_random") & (grid_rows.noise_mode == "per_cell")
                   & (grid_rows.noise_sigma_frac == 0.02) & (grid_rows.stage == "headline")]  # fmt: skip
    for (ck, ly, T, enc), g in mx.groupby(["ckpt_seed", "layer", "tile_size", "encoding"]):
        per_k = g.groupby("k").rel_shift_median.median()
        if len(per_k) >= 3:
            fits.append({"row_type": "h12_info", "ckpt_seed": ck, "layer": ly, "tile_size": T,
                         "encoding": enc, "info_h12_slope_k_mixed_rail": _loglog_slope(per_k.index, per_k.values)})  # fmt: skip
    out.append(pd.DataFrame(fits))

    # ---- H1.4: is the H1.1 conclusion the same in every encoding?
    h11 = cells[(cells.k == 1) & (cells.noise_mode == "per_cell") & (cells.noise_sigma_frac == 0.02)
                & cells.sigma1_auc.notna()]  # fmt: skip
    fr = h11.groupby("encoding").sigma1_auc.apply(lambda s: float((s < 0.6).mean()))
    info["h11_frac_auc_below_0.6_by_encoding"] = fr.to_dict()
    outcomes = {e: bool(v >= 0.95) for e, v in fr.items()}
    info["h11_outcome_by_encoding"] = outcomes
    out.append(pd.DataFrame([{"row_type": "h14", "h14_h11_conclusion_identical":
                              int(len(set(outcomes.values())) == 1)}]))  # fmt: skip
    return pd.concat(out, ignore_index=True), info


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "f0_1.yaml", "f0_1", argv)
    cfg = load_config(a.config, a.smoke)
    cfg["master_seed"] = a.seed
    grid = build_grid(cfg)
    print(f"F0.1 [{cfg['_variant']}]: {len(grid)} tasks, jobs={a.jobs}", flush=True)
    tile_df = run_grid(task, grid, jobs=a.jobs, out_dir=Path(a.out) / "_cells_cache")
    metrics, info = aggregate(tile_df)
    pd.set_option("display.width", 220)
    print("\nH1.1 fraction of cells with sigma_1 AUC < 0.6 (k=1, 2% per_cell), by encoding:")
    print(info["h11_frac_auc_below_0.6_by_encoding"])
    h12 = metrics[metrics.row_type == "h12"]
    for col in ("h12_slope_tile", "h12_slope_k", "info_h12_slope_tile_geom_all_shapes",
                "info_h12_slope_tile_nominal_all_shapes"):  # fmt: skip
        if col in h12 and h12[col].notna().any():
            v = h12[col].dropna()
            print(f"{col}: median {v.median():.3f}  IQR [{v.quantile(.25):.3f}, "
                  f"{v.quantile(.75):.3f}]  n={len(v)}")  # fmt: skip
    finish("F0.1", "f0_stuck_scaling.yaml", cfg, a, metrics, extra=info,
                        side_tables={"tile_metrics": tile_df})  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

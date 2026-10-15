# ruff: noqa: C408
"""Phase H / H7 (E5): concurrent fault detection + localization from live inputs (criteria/b_concurrent.yaml).

Two analog checksum lines per tile (sum and index-weighted, Neural Checksums / HPE style); every inference MVM gives
two syndromes. CK = checksum detection, HPE-VOTE = weighted-checksum row vote, A8 = OMP regression of the syndromes
on the live inputs (ours). Dedicated comparators: stored natural test inputs (GT) vs random +-1 probes (RP).

  python experiments/spectral/b_concurrent.py --jobs 2 --allow-dirty [--smoke]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
from a_common import CONV_LAYERS, ENC, PROG, act_stats, select_tiles
from f0_common import finish, get_tile, load_config, parse_args
from scipy.stats import norm

from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.hardening import interleave_index, merge_rows
from neurofault.spectral.harness import run_grid
from neurofault.spectral.noise import program
from neurofault.spectral.readout import TileReader
from neurofault.spectral.types import DTYPE, FaultSpec, ReadNoise, Tile, make_generator

N_COMM = 2048
N_PASS = 128  # commissioning read passes over the 2048 commissioning inputs (262k MVMs, once after programming)
Z1 = float(norm.ppf(1 - 0.005))  # per-sample two-sided 1% on whitened residuals
SEED_OFFSET = 5000


def _syndromes(
    y: torch.Tensor, arr_idx, ref_idx, m: int, gain
) -> tuple[torch.Tensor, torch.Tensor]:
    """gain = (g1, gw): each checksum line is programmed at full conductance range, c / g, and read back x g."""
    Y = y[arr_idx]
    wts = (torch.arange(m, dtype=DTYPE) + 1) / m
    return Y.sum(0) - gain[0] * y[ref_idx[0]], wts @ Y - gain[1] * y[ref_idx[1]]


def _ridge(X: torch.Tensor, s: torch.Tensor, lam: float = 1e-7) -> torch.Tensor:
    """d minimizing ||s - d X||^2 + lam ||X||_F^2/n ||d||^2 (X: n x T, s: T)."""
    n = X.shape[0]
    G = X @ X.T
    G = G + lam * torch.trace(G) / n * torch.eye(n, dtype=DTYPE)
    return torch.linalg.solve(G, X @ s)


def omp_support(r: torch.Tensor, Xw: torch.Tensor, z_stop: float, max_atoms: int = 64):
    """OMP of r (T) on rows of Xw (n x T, whitened). Returns (support list, coefficients)."""
    support: list[int] = []
    coef = torch.zeros(0, dtype=DTYPE)
    res = r.clone()
    norms = Xw.norm(dim=1).clamp_min(1e-12)
    for _ in range(max_atoms):
        z = (Xw @ res) / norms
        if support:
            z[support] = 0.0
        j = int(z.abs().argmax())
        if float(z[j].abs()) < z_stop:
            break
        support.append(j)
        A = Xw[support].T
        coef = torch.linalg.lstsq(A, r[:, None]).solution[:, 0]
        res = r - A @ coef
    return support, coef


def _f1(pred: set, truth: set) -> float:
    if not truth:
        return float("nan")
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, rc = tp / len(pred), tp / len(truth)
    return 2 * p * rc / (p + rc)


def tile_job(p: dict) -> list[dict]:
    torch.set_num_threads(1)
    cfg, (seed, layer, r0, c0, m, n, size) = p["cfg"], p["tile"]
    rs = seed + SEED_OFFSET
    tag = f"{layer}:{r0}:{c0}"
    tile = get_tile(seed, layer, r0, c0, m, n, size)
    w0 = tile.w.to(DTYPE)
    wmax = float(w0.abs().max())
    absmax = wmax
    st0 = encode(tile, ENC, absmax=absmax)
    stp = program(st0, PROG, make_generator(rs, f"h7prog:{tag}"))
    wts = (torch.arange(m, dtype=DTYPE) + 1) / m
    c1, cw = w0.sum(0), wts @ w0
    gain = (float(c1.abs().max()) / wmax, float(cw.abs().max()) / wmax)
    wref = torch.stack([c1 / gain[0], cw / gain[1]])
    ref_stp = program(
        encode(Tile(tile.spec, wref), ENC, absmax=absmax), PROG, make_generator(rs, f"h7ref:{tag}")
    )
    arr_idx, ref_idx, _ = interleave_index(m, 2)
    noise = ReadNoise("per_cell", cfg["sigma"], adc_bits=None, dac_bits=None)
    X = act_stats(seed, layer)["X_cache"][c0 : c0 + n].to(DTYPE)
    Xc, Xm = X[:, :N_COMM], X[:, N_COMM:]
    xnorm2_m = Xm.pow(2).sum(0)
    dead_frac = float(((X != 0).float().mean(1) < 0.01).float().mean())
    z_stop = float(norm.ppf(1 - 0.005 / n))
    Tg = list(cfg["T_grid"])
    T_max = max(Tg)
    acts = act_stats(seed, layer)
    mvm_per_image = acts["positions_total"] / acts["n_images"]

    # commissioning on the healthy programmed tile (fault-free): N_PASS read passes over the 2048 commissioning inputs
    # fit the mismatch vectors d (programming error of the checksum lines); one more pass, never used in the fit,
    # estimates the noise scale out of sample
    healthy = merge_rows(stp, ref_stp, arr_idx, ref_idx)
    rd = TileReader(lambda: healthy, noise, make_generator(rs, f"h7comm:{tag}"))
    s1c = torch.zeros(N_COMM, dtype=DTYPE)
    swc = torch.zeros(N_COMM, dtype=DTYPE)
    for _ in range(
        N_PASS
    ):  # same inputs each pass: averaging the syndromes = LS on the repeated stream
        a1, aw = _syndromes(rd.mvm(Xc), arr_idx, ref_idx, m, gain)
        s1c += a1 / N_PASS
        swc += aw / N_PASS
    d1, dw = _ridge(Xc, s1c), _ridge(Xc, swc)
    s1v, swv = _syndromes(rd.mvm(Xc), arr_idx, ref_idx, m, gain)
    xn2c = Xc.pow(2).sum(0).clamp_min(1e-12)
    v1 = float(((s1v - d1 @ Xc).pow(2) / xn2c).mean())
    vw = float(((swv - dw @ Xc).pow(2) / xn2c).mean())
    c1w = float(
        ((s1v - d1 @ Xc) * (swv - dw @ Xc) / xn2c).mean()
    )  # shared read noise of the two syndromes
    # stored references for the dedicated comparators (measured fault-free outputs)
    P_max = max(cfg["P_grid"])
    gi = torch.randperm(N_COMM, generator=make_generator(rs, f"h7gt:{tag}"))[:P_max]
    Xgt = Xc[:, gi]
    Xrp = torch.where(
        torch.rand(n, P_max, generator=make_generator(rs, f"h7rp:{tag}")) < 0.5, -1.0, 1.0
    ).to(DTYPE)
    ref_gt = rd.mvm(Xgt)[arr_idx]
    ref_rp = rd.mvm(Xrp)[arr_idx]

    rows = []
    for k, n_inst in cfg["instances"].items():
        k = int(k)
        for fi in range(n_inst):
            g = make_generator(rs, f"h7fault:{tag}:{k}:{fi}")
            if k > 0:
                bad, truth = inject(
                    stp, FaultSpec("stuck_random", k=k, rail="mixed", device="either"), g
                )
                vis = truth.delta_w.abs() >= 0.1 * wmax
                cells = truth.cells[vis]
                big = bool(vis.any() and float(truth.delta_w.abs().max()) >= 0.5 * wmax)
                t_cells = {(int(a), int(b)) for a, b in cells.tolist()}
            else:
                bad, t_cells, big = stp, set(), False
            if k > 0 and not t_cells:
                continue
            aug = merge_rows(bad, ref_stp, arr_idx, ref_idx)
            rd_i = TileReader(lambda a=aug: a, noise, make_generator(rs, f"h7read:{tag}:{k}:{fi}"))
            # monitoring stream: T_max samples drawn cyclically from the 2048 held-out live inputs (each pass in a
            # fresh random order); every read has fresh noise
            go = make_generator(rs, f"h7ord:{tag}:{k}:{fi}")
            reps = -(-T_max // Xm.shape[1])
            order = torch.cat([torch.randperm(Xm.shape[1], generator=go) for _ in range(reps)])[
                :T_max
            ]
            Xo, xn2 = Xm[:, order], xnorm2_m[order].clamp_min(1e-12)
            s1, sw = _syndromes(rd_i.mvm(Xo), arr_idx, ref_idx, m, gain)
            sc1, scw = (v1 * xn2).sqrt(), (vw * xn2).sqrt()
            r1 = (s1 - d1 @ Xo) / sc1
            rw = (sw - dw @ Xo) / scw
            Xw = Xo / sc1[None, :]
            cs = torch.cumsum(r1.pow(2), 0)
            base = dict(ckpt_seed=seed, layer=layer, r0=r0, c0=c0, m=m, n=n, k=k, inst=fi, big=big,
                        n_vis=len(t_cells), dead_frac=dead_frac, mvm_per_image=mvm_per_image)  # fmt: skip
            t_rows = {i for i, _ in t_cells}
            t_cols = {j for _, j in t_cells}
            for T in Tg:
                rec = dict(base, T=T, S=float(cs[T - 1]))
                flags = r1[:T].abs() > Z1
                if k == 0:
                    rec["fa_sample"] = float(flags.float().mean())
                if k > 0:
                    # HPE-VOTE: flagged samples vote for row round(m * sw / s1) - 1 (unwhitened residuals)
                    rr1, rrw = (r1[:T] * sc1[:T])[flags], (rw[:T] * scw[:T])[flags]
                    if len(rr1):
                        votes = torch.round(m * rrw / rr1).sub(1).clamp(0, m - 1).long()
                        cnt = torch.bincount(votes, minlength=m)
                        pred_rows = set(
                            torch.nonzero(cnt >= max(3, 0.1 * len(votes))).flatten().tolist()
                        )
                    else:
                        pred_rows = set()
                    rec["hpe_vote_line_f1"] = _f1(pred_rows, t_rows)
                    # HPE (registered): single-error decode with temporal averaging - LS ratio over flagged samples
                    if len(rr1):
                        # moment-corrected for the noise variance of s1 and the s1/sw noise covariance
                        xf = xn2[:T][flags]
                        num = float((rrw * rr1 - c1w * xf).sum())
                        den = float((rr1.pow(2) - v1 * xf).sum())
                        ratio = num / den if den > 0 else 0.0
                        i_hat = min(max(round(m * ratio) - 1, 0), m - 1)
                        rec["hpe_line_f1"] = _f1({i_hat}, t_rows)
                    else:
                        rec["hpe_line_f1"] = 0.0
                    rec["hpe_n_flag"] = int(flags.sum())
                sup, a = omp_support(r1[:T], Xw[:, :T], z_stop)  # A8 (also on controls: its FA)
                rec["a8_alarm"] = bool(sup)
                if k > 0:
                    pred_cells = set()
                    if sup:
                        Xs = (Xo[sup, :T] / scw[None, :T]).T
                        b = torch.linalg.lstsq(Xs, rw[:T, None]).solution[:, 0]
                        # a, b are in unwhitened syndrome units: b_j / a_j = (i + 1) / m
                        for j, aj, bj in zip(sup, a.tolist(), b.tolist(), strict=True):
                            i = round(m * bj / aj) - 1 if abs(aj) > 1e-12 else 0
                            pred_cells.add((min(max(i, 0), m - 1), j))
                    rec["a8_cell_f1"] = _f1(pred_cells, t_cells)
                    # A8 names INPUT lines (columns); the row from b/a is coarse (report only)
                    rec["a8_line_f1"] = _f1({j for _, j in pred_cells}, t_cols)
                    rec["a8_row_f1"] = _f1({i for i, _ in pred_cells}, t_rows)
                    rec["a8_n_pred"] = len(pred_cells)
                rows.append(rec)
            # dedicated comparators: per-row chi2 of P test reads vs stored measured reference (max over rows,
            # the GLRT for a sparse fault set)
            y_gt = rd_i.mvm(Xgt)[arr_idx] - ref_gt
            y_rp = rd_i.mvm(Xrp)[arr_idx] - ref_rp
            for P in cfg["P_grid"]:
                for meth, yy, xx in (("gt", y_gt, Xgt), ("rp", y_rp, Xrp)):
                    q = yy[:, :P].pow(2) / (xx[:, :P].pow(2).sum(0).clamp_min(1e-12) * wmax**2)
                    stat = float(q.sum(1).max())
                    rows.append(dict(base, T=-1, dedicated=meth, P=P, stat=stat))
    return rows


def build_metrics(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    conc = raw[raw["T"] > 0].copy()
    ded = raw[raw["T"] < 0].copy()
    out = []
    ctrl = conc[conc.k == 0]
    thr = ctrl.groupby("T").S.quantile(0.99)
    fa = {T: float((ctrl[ctrl["T"] == T].S > thr[T]).mean()) for T in thr.index}
    for seed in sorted(conc.ckpt_seed.unique()):
        cs = conc[conc.ckpt_seed == seed]
        for T in sorted(cs["T"].unique()):
            d = cs[(cs["T"] == T) & (cs.k == 1) & cs.big]
            cc = ctrl[(ctrl["T"] == T) & (ctrl.ckpt_seed == seed)]
            out.append(dict(kind="detect", ckpt_seed=seed, T=T,
                            det_prob_big_k1=float(d.a8_alarm.mean()), fa_a8=float(cc.a8_alarm.mean()),
                            det_prob_big_k1_ck=float((d.S > thr[T]).mean()), fa_ck=fa[T], thr_ck=float(thr[T]),
                            n=len(d), images_equiv=float((T / cs[cs["T"] == T].mvm_per_image).median()),
                            fa_sample_ctrl=float(cc.fa_sample.mean())))  # fmt: skip
            for k in sorted(cs.k.unique()):
                if k == 0:
                    continue
                dk = cs[(cs["T"] == T) & (cs.k == k)]
                out.append(dict(kind="detect_all", ckpt_seed=seed, T=T, k=k, det_prob_a8=float(dk.a8_alarm.mean()),
                                det_prob_ck=float((dk.S > thr[T]).mean()), n=len(dk)))  # fmt: skip
                a8 = float(dk.a8_line_f1.mean())
                hp = float(dk.hpe_line_f1.mean())
                out.append(dict(kind="locate", ckpt_seed=seed, T=T, k=k, method="a8", line_f1=a8,
                                cell_f1=float(dk.a8_cell_f1.mean()), row_f1=float(dk.a8_row_f1.mean()),
                                n_pred=float(dk.a8_n_pred.mean()), n=len(dk)))  # fmt: skip
                out.append(
                    dict(
                        kind="locate", ckpt_seed=seed, T=T, k=k, method="hpe", line_f1=hp, n=len(dk)
                    )
                )
                out.append(dict(kind="locate", ckpt_seed=seed, T=T, k=k, method="hpe_vote",
                                line_f1=float(dk.hpe_vote_line_f1.mean()), n=len(dk)))  # fmt: skip
                out.append(
                    dict(kind="locate_diff", ckpt_seed=seed, T=T, k=k, a8_minus_hpe_line_f1=a8 - hp)
                )
    tiles = conc.drop_duplicates(["ckpt_seed", "layer", "r0", "c0"])
    for _, r in tiles.iterrows():
        out.append(
            dict(
                kind="ident",
                ckpt_seed=r.ckpt_seed,
                layer=r.layer,
                r0=r.r0,
                c0=r.c0,
                dead_frac=r.dead_frac,
            )
        )
    if len(ded):
        for (meth, P), g in ded.groupby(["dedicated", "P"]):
            t = g[g.k == 0].stat.quantile(0.99)
            for k in sorted(g.k.unique()):
                if k == 0:
                    continue
                out.append(dict(kind="dedicated", method=meth, P=P, k=k,
                                det_power=float((g[g.k == k].stat > t).mean()), n=int((g.k == k).sum())))  # fmt: skip
    return pd.DataFrame(out)


def main(argv=None) -> int:
    a = parse_args(__doc__.splitlines()[0], "b_concurrent.yaml", "b_concurrent", argv)
    cfg = load_config(a.config, a.smoke)
    cfg.update(master_seed=a.seed)
    tiles = select_tiles(cfg["ckpt_seeds"], cfg.get("layers", CONV_LAYERS), cfg["tiles_per_layer"])
    pts = [dict(cfg=cfg, tile=t) for t in tiles]
    raw = run_grid(tile_job, pts, jobs=a.jobs, out_dir=Path(a.out) / "_cells")
    df = build_metrics(raw, cfg)
    pd.set_option("display.width", 250)
    loc = df[df.kind == "locate"].pivot_table(
        index=["k", "T"], columns="method", values="line_f1", aggfunc="mean"
    )
    print(loc.round(3).to_string())
    print(
        df[df.kind == "detect"]
        .groupby("T")[["det_prob_big_k1", "fa_a8", "det_prob_big_k1_ck", "fa_ck", "images_equiv"]]
        .mean()
        .round(3)
        .to_string()
    )
    print(
        df[df.kind == "dedicated"]
        .pivot_table(index=["k", "P"], columns="method", values="det_power")
        .round(3)
        .to_string()
    )
    print("dead_frac median:", float(df[df.kind == "ident"].dead_frac.median()))
    slim = raw
    run_dir, _ = finish("B-concurrent H7", "b_concurrent.yaml", cfg, a, df,
                        extra={"note": "H7 concurrent detection/localization (E5)"},
                        side_tables=dict(raw=slim))  # fmt: skip
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

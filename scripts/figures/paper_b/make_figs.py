"""Paper B figures. Reads committed CSVs under results/spectral only.

Run:  .venv/bin/python scripts/figures/paper_b/make_figs.py
Writes PDFs to papers/spectral/figs/. The mechanism diagram fig_bidirectional.pdf is built
from scripts/figures/paper_b/fig_bidirectional.tex (TikZ standalone, tectonic).
Palette order: blue, orange, aqua, amber, pink; every series also has its own marker and line style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "papers" / "spectral" / "figs"
RES = ROOT / "results" / "spectral"
SRC = {
    "bnfold": RES / "a_bnfold_v2" / "20261010T022159_fca882d8_s0",
    "gate0": RES / "a_gate0" / "20261009T020655_ae67b976_s0",
    "exp2": RES / "a_exp2" / "20261009T052819_52891c48_s0",
    "c1b": RES / "b_c1b" / "20261009T115059_609a698a_s0",
    "f01": RES / "f0_1" / "20261009T023921_a8f8370e_s0",
}
BLUE, ORANGE, AQUA, AMBER, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
DGRAY = "#444444"
# arm -> (colour, marker, linestyle); the same arm looks the same in every figure.
ARM = {
    "m4": (BLUE, "o", "-"),
    "b8": (ORANGE, "v", "--"),
    "ada": (AQUA, "^", "-."),
    "m4ada": (AMBER, "D", ":"),
    "gdc": (PINK, "s", "-"),
    "prog": (DGRAY, "", "--"),
}
COL = 3.5
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    }
)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_drift_accuracy():
    """Accuracy vs time: programmed, M4 (strict 16 MVM), B8. Points = checkpoint seeds, line = mean."""
    s = pd.read_csv(SRC["bnfold"] / "summary_strict.csv")
    raw = pd.read_csv(SRC["bnfold"] / "metrics_B_strict16.csv")
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.0), sharex=True)
    for ax, ds, title in zip(axs, ["cifar10", "cifar100"], ["CIFAR-10", "CIFAR-100"]):
        d = s[s.dataset == ds].sort_values("t_s")
        r = (
            raw[raw.dataset == ds]
            .groupby(["ckpt_seed", "t_s"])[["acc_m4", "acc_b8"]]
            .mean()
            .reset_index()
        )
        for key, col in (("m4", "acc_m4"), ("b8", "acc_b8")):
            c, m, _ls = ARM[key]
            ax.plot(r.t_s, r[col], ls="none", marker=m, ms=2.5, color=c, alpha=0.35)
        ax.plot(d.t_s, d.acc_prog, color=DGRAY, ls="--", lw=1)
        ax.plot(d.t_s, d.acc_m4, color=BLUE, marker="o", ms=4, lw=1.2, ls="-")
        ax.plot(d.t_s, d.acc_b8, color=ORANGE, marker="v", ms=4, lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([3600, 86400, 2592000, 31557600])
        ax.set_xticklabels(["1h", "1d", "30d", "1y"])
        ax.set_title(title)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[0].set_ylabel("top-1 accuracy (%)")
    fig.supxlabel("time since programming", fontsize=8, y=-0.07)
    h = [
        Line2D([], [], color=DGRAY, ls="--", label="programmed"),
        Line2D([], [], color=BLUE, marker="o", ms=4, label="M4 (16 MVM)"),
        Line2D([], [], color=ORANGE, marker="v", ms=4, ls="--", label="best cheap (B8)"),
    ]
    axs[0].legend(handles=h, loc="lower left")
    save(fig, "fig_drift_accuracy.pdf")


def fig_gate0():
    """Gate 0': output error of B8 vs M4 as pattern strength rho grows (read noise 2%)."""
    d = pd.read_csv(SRC["gate0"] / "incumbent_vs_m4.csv")
    d = d[(d.sigma == 0.02) & (d.model == "D3")]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    styles = {"hotspot": "-", "rank_r": "--", "row_gradient": ":"}
    for fld, ls in styles.items():
        g = d[d.field == fld].sort_values("rho")
        ax.plot(g.rho, 100 * g.err_b8, color=ORANGE, ls=ls, marker="v", ms=3.5)
        ax.plot(g.rho, 100 * g.err_m4best, color=BLUE, ls=ls, marker="o", ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"pattern strength $\rho$")
    ax.set_ylabel("rel. output error (%)")
    h = [
        Line2D([], [], color=ORANGE, marker="v", ms=3.5, label="B8 (best cheap)"),
        Line2D([], [], color=BLUE, marker="o", ms=3.5, label="M4"),
    ]
    h += [Line2D([], [], color="k", ls=ls, label=f.replace("_", " ")) for f, ls in styles.items()]
    ax.legend(handles=h, loc="upper left")
    save(fig, "fig_gate0.pdf")


def fig_exp2_pareto():
    """Exp 2: accuracy vs calibration cost (MVM-equivalents), BN kept and BN folded."""
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.0))
    arm_of = {
        "m4_16": "m4",
        "b8": "b8",
        "gdc": "gdc",
        "ada256": "ada",
        "ada2600": "ada",
        "m4ada_16": "m4ada",
    }
    for ax, wk, title in zip(axs, ["W1", "W2"], ["BN kept", "BN folded"]):
        d = pd.read_csv(SRC["exp2"] / f"pareto_{wk}.csv")
        orc = float(d[d.config == "oracle"].acc.iloc[0])
        ax.axhline(orc, color=DGRAY, ls="--", lw=0.9)
        for _, r in d.iterrows():
            if r.config not in arm_of or r.cost_mvm <= 0:
                continue
            c, m, _ = ARM[arm_of[r.config]]
            ax.plot(r.cost_mvm, r.acc, ls="none", marker=m, color=c, ms=4.5)
            right = r.config in ("ada2600", "m4ada_16", "ada256")
            ax.annotate(
                r.config.replace("_", "-"),
                (r.cost_mvm, r.acc),
                fontsize=6,
                xytext=(-3 if right else 3, 3 if r.config != "b8" else -7),
                ha="right" if right else "left",
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_xlim(4e2, 3e8)
        ax.set_xticks([1e3, 1e5, 1e7])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title)
    fig.supxlabel("calibration cost (MVM-equivalents)", fontsize=8, y=-0.07)
    axs[0].set_ylabel("top-1 accuracy (%)")
    save(fig, "fig_exp2_pareto.pdf")


def fig_reads_ratio():
    """C1b: reads(bidirectional) / reads(best forward-only) at F1>=0.9, by k, tile size, noise."""
    d = pd.read_csv(SRC["c1b"] / "verdict_metrics.csv")
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True)
    tiles = {128: (BLUE, "o", "-"), 256: (ORANGE, "s", "--"), 512: (AQUA, "^", "-.")}
    for ax, sg in zip(axs, [0.01, 0.02]):
        for ts, (c, m, ls) in tiles.items():
            g = d[(d.sigma_c == sg) & (d.tile_size == ts)].sort_values("k")
            ax.plot(
                g.k, g.reads_ratio, marker=m, ms=3.5, lw=1.1, ls=ls, color=c, label=f"tile {ts}"
            )
        ax.axhline(0.7, color="k", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("faults $k$")
        ax.set_title(rf"read noise $\sigma_c={int(sg * 100)}\%$")
    axs[0].set_ylabel("reads ratio (bidir / fwd)")
    axs[0].set_ylim(0, 1.08)
    axs[1].legend(loc="lower right")
    save(fig, "fig_reads_ratio.pdf")


def fig_sigma1_auc():
    """F0.1: AUC of the sigma_1 shift as a stuck-cell detector vs tile size and k (2% noise)."""
    d = pd.read_csv(SRC["f01"] / "tile_metrics.csv")
    h = d[
        (d.stage == "headline")
        & (d.fault_kind == "stuck_random")
        & (d.noise_mode == "per_cell")
        & (d.noise_sigma_frac == 0.02)
        & (d.encoding == "signed_ideal")
    ]
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ser = [(1, BLUE, "o", "-"), (4, ORANGE, "s", "--"), (16, AQUA, "^", "-."), (64, PINK, "D", ":")]
    for k, c, m, ls in ser:
        g = h[h.k == k].groupby("tile_size").sigma1_auc.mean()
        ax.plot(g.index, g.values, marker=m, ms=3.5, ls=ls, color=c, label=f"$\\sigma_1$, $k$={k}")
    p = h[h.k == 1].groupby("tile_size").probe_p8_auc.mean()
    ax.plot(
        p.index,
        p.values,
        color="k",
        ls="-",
        marker="x",
        ms=4,
        lw=1.0,
        label=r"8 probe reads, $k$=1",
    )
    ax.axhline(0.5, color=DGRAY, lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512])
    ax.set_xticklabels(["64", "128", "256", "512"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("tile size")
    ax.set_ylabel(r"AUC of $\sigma_1$ shift")
    ax.legend(ncol=2, loc="upper right", fontsize=6.5)
    save(fig, "fig_sigma1_auc.pdf")


if __name__ == "__main__":
    fig_drift_accuracy()
    fig_gate0()
    fig_exp2_pareto()
    fig_reads_ratio()
    fig_sigma1_auc()

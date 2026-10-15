"""Fig 3: final test accuracy per arm, seeds as points plus mean (ResNet-9, 3x10% waves, 15% spares)."""

import _common as c
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ORDER = ["R0", "R1", "E2", "E1F", "B", "BM", "E1L", "C'M", "D"]
SHORT = {
    "R0": "R0 (no mitigation)",
    "R1": "R1 (remap only)",
    "E2": "E2 (BN recal only)",
    "E1F": "E1-first",
    "B": "B (periodic)",
    "BM": "B-matched",
    "E1L": "E1-last (Liu-style)",
    "C'M": "C'-matched",
    "D": "D (controller)",
}


def main() -> None:
    c.setup()
    data = {k: c.arm(c.MAIN[k]) for k in ORDER}
    nmax = max(len(v) for v in data.values())
    fig = plt.figure(figsize=(c.PAGE, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1, 1.9], wspace=0.04, figure=fig)
    axl, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharey=None)
    axl.set_xlim(5, 30)
    axr.set_xlim(79, 90)
    labels = []
    for i, k in enumerate(ORDER):
        v = data[k]
        n = len(v)
        labels.append(SHORT[k] + (f"  [partial (n={n})]" if n < nmax else f"  [n={n}]"))
        col, mk, _, _ = c.ARM[k]
        seeds = sorted(v)
        print(f"{k}: n={n} mean={c.mean(v.values()):.2f} seeds {c.fmtv(v)}")
        for ax in (axl, axr):
            ys = [i + c.jitter(j, n, 0.2) for j in range(n)]
            ax.scatter(
                [v[s] for s in seeds],
                ys,
                s=22 if mk in "X*" else 14,
                color=col,
                marker=mk,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
                alpha=0.9,
            )
            m = c.mean(v.values())
            ax.plot([m, m], [i - 0.34, i + 0.34], color="black", lw=1.6, zorder=4)
    for ax in (axl, axr):
        ax.set_ylim(len(ORDER) - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
    axl.set_yticks(range(len(ORDER)), labels)
    axr.set_yticks([])
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axl.set_xticks([10, 20, 30])
    axr.set_xticks([80, 82, 84, 86, 88, 90])
    axr.spines["right"].set_visible(False)
    d = 0.012
    for ax, x in ((axl, 1), (axr, 0)):
        kw = {"transform": ax.transAxes, "color": "k", "clip_on": False, "lw": 0.7}
        ax.plot((x - 0.02 * (1 + 0.9 * (x == 0)), x + 0.02), (-d * 2, d * 2), **kw)
    fig.supxlabel(
        "Final test accuracy (%); axis broken between 30 and 79. Black bar = mean, markers = seeds",
        fontsize=8,
        y=-0.02,
    )
    c.save(fig, "fig3_main.pdf")


if __name__ == "__main__":
    main()

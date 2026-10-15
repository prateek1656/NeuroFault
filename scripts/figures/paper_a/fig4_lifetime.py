"""Fig 4: per-wave accuracy, mean line plus min-max band, D vs C'-matched vs B-matched vs R1.

One y quantity (test accuracy) drawn on a vertically broken axis (top 78-90, bottom 0-20)
so the three retraining arms stay separable while R1's collapse is still shown.
"""

import _common as c
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

KEYS = ["D", "C'M", "BM", "R1"]


def main() -> None:
    c.setup()
    fig = plt.figure(figsize=(c.COL + 0.7, 3.3))
    gs = GridSpec(2, 1, height_ratios=[3.4, 1], hspace=0.07, figure=fig)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    clean = c.arm(c.MAIN["D"], "baseline_test_acc")
    for k in KEYS:
        col, mk, ls, lab = c.ARM[k]
        w = c.waves(c.MAIN[k])
        seeds = sorted(w)
        n_w = len(w[seeds[0]])
        x = list(range(1, n_w + 1))
        mn = [min(w[s][i] for s in seeds) for i in range(n_w)]
        mx = [max(w[s][i] for s in seeds) for i in range(n_w)]
        m = [c.mean(w[s][i] for s in seeds) for i in range(n_w)]
        if k == "R1":  # wave 1 in the top panel, waves 2-3 in the bottom panel
            parts = ((top, slice(0, 1)), (bot, slice(1, None)))
        else:
            parts = ((top, slice(0, None)),)
        for ax, sl in parts:
            ax.fill_between(x[sl], mn[sl], mx[sl], color=col, alpha=0.2, lw=0)
            ax.plot(
                x[sl],
                m[sl],
                color=col,
                ls=ls,
                marker=mk,
                ms=5,
                lw=1.5,
                label=f"{lab} (n={len(seeds)})" if ax is top else None,
            )
        print(
            f"{k}: n={len(seeds)} mean per wave {[round(v, 2) for v in m]} "
            f"min {[round(v, 2) for v in mn]} max {[round(v, 2) for v in mx]}"
        )
    cm = c.mean(clean.values())
    top.axhline(cm, color=c.GRAY, lw=0.8, ls=(0, (1, 2)))
    top.text(1.97, cm + 0.35, f"clean chip {cm:.1f}", color=c.DGRAY, fontsize=7.5, ha="right")
    print(f"clean mean (D seeds) {cm:.2f}")
    top.set_ylim(78, 90.5)
    bot.set_ylim(0, 20)
    bot.set_yticks([0, 10, 20])
    top.set_xlim(0.8, 3.2)
    bot.set_xlim(0.8, 3.2)
    top.set_xticks([])
    top.spines["bottom"].set_visible(False)
    bot.set_xticks([1, 2, 3])
    bot.set_xlabel("Fault wave (10% newly stuck each)")
    fig.supylabel("Test accuracy after wave (%)", fontsize=8.5, x=-0.01)
    bot.text(3.15, 14.5, "R1 (waves 2-3)", fontsize=7.5, color=c.DGRAY, ha="right")
    top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7,
        columnspacing=1.0,
        handlelength=2.4,
    )
    c.save(fig, "fig4_lifetime.pdf")


if __name__ == "__main__":
    main()

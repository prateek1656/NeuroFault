"""Fig 7: burst-rate tuning changed the headline. Old rate 0.005 vs tuned rate, paired by seed."""

import _common as c
import matplotlib.pyplot as plt

PAIRS = [  # key, tuned spec, old spec, tuned rate
    ("B", "resnet9_deploy_b_eta5e-1", "resnet9_deploy_b", "0.5"),
    ("C'", "resnet9_deploy_c_living_eta5e-1", "resnet9_deploy_c_living", "0.5"),
    ("Liu", "resnet9_deploy_liu_style_lr0.5", "resnet9_deploy_liu_style_lr0.005", "0.5"),
    ("D", "resnet9_deploy_d_eta3e-1", "resnet9_deploy_d", "0.3"),
]


def main() -> None:
    c.setup()
    fig, ax = plt.subplots(figsize=(c.COL + 1.2, 3.0))
    means = {}
    for i, (k, tuned, old, rate) in enumerate(PAIRS):
        col, mk, _, lab = c.ARM[k]
        t, o = c.arm(tuned), c.arm(old)
        seeds = sorted(set(t) & set(o))
        xo, xt = i * 3.0, i * 3.0 + 1.1
        for s in seeds:
            ax.plot([xo, xt], [o[s], t[s]], color=col, lw=0.6, alpha=0.6)
        ax.scatter(
            [xo] * len(seeds),
            [o[s] for s in seeds],
            s=16,
            marker=mk,
            facecolor="white",
            edgecolor=col,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter([xt] * len(seeds), [t[s] for s in seeds], s=16, marker=mk, color=col, zorder=3)
        mo, mt = c.mean(o[s] for s in seeds), c.mean(t[s] for s in seeds)
        means[k] = (mo, mt)
        ax.plot([xo - 0.3, xo + 0.3], [mo, mo], color="black", lw=1.5, zorder=4)
        ax.plot([xt - 0.3, xt + 0.3], [mt, mt], color="black", lw=1.5, zorder=4)
        ax.text(xo - 0.35, mo, f"{mo:.1f}", ha="right", va="center", fontsize=7.5)
        ax.text(xt + 0.35, mt, f"{mt:.1f}", ha="left", va="center", fontsize=7.5)
        ax.text(
            (xo + xt) / 2,
            2.5,
            lab.split(" (")[0],
            ha="center",
            color=col,
            fontsize=8,
            weight="bold",
        )
        print(
            f"{k}: n={len(seeds)} old(0.005) mean {mo:.2f} seeds {c.fmtv(o)} | "
            f"tuned({rate}) mean {mt:.2f} seeds {c.fmtv(t)} | gain {mt - mo:+.2f}"
        )
    gap_old = means["D"][0] - means["C'"][0]
    gap_new = means["D"][1] - means["C'"][1]
    print(f"D minus C' unmatched: old {gap_old:+.2f}, tuned {gap_new:+.2f}")
    ax.text(
        0.5,
        0.965,
        f"D minus C': {gap_old:+.1f} at rate 0.005\n{gap_new:+.1f} after tuning",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": c.GRAY, "lw": 0.6},
    )
    ax.scatter([], [], s=16, marker="o", facecolor="white", edgecolor="black", label="rate 0.005")
    ax.scatter([], [], s=16, marker="o", color="black", label="tuned rate (B, C', Liu 0.5; D 0.3)")
    ax.set_xticks([])
    ax.set_xlim(-1.2, 3 * 3.0 + 2.4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Final test accuracy (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=7)
    ax.grid(axis="x", visible=False)
    c.save(fig, "fig7_tuning.pdf")


if __name__ == "__main__":
    main()

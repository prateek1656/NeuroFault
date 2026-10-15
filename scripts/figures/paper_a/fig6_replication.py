"""Fig 6: per-seed D minus C'-matched; original seeds vs pre-registered E3 replication seeds."""

import _common as c
import matplotlib.pyplot as plt

ORIG = (42, 101, 102, 103, 104)
E3 = (105, 106, 107)


def main() -> None:
    c.setup()
    d, cm = c.arm(c.MAIN["D"]), c.arm(c.MAIN["C'M"])
    steps = {s: r["total_optimizer_steps"] for s, r in c.rows(c.MAIN["D"]).items()}
    gap = {s: d[s] - cm[s] for s in d if s in cm}
    fig, ax = plt.subplots(figsize=(c.COL + 0.9, 2.9))
    allseeds = list(ORIG) + list(E3)
    pos = {s: i + (0.8 if s in E3 else 0) for i, s in enumerate(allseeds)}
    ax.axhline(0, color="black", lw=0.9)
    cols = {"orig": (c.BLUE, "o"), "e3": (c.ORANGE, "s")}
    for grp, seeds, key in (
        ("orig", ORIG, "original (42, 101-104)"),
        ("e3", E3, "pre-registered replication (105-107)"),
    ):
        col, mk = cols[grp]
        got = [s for s in seeds if s in gap]
        ax.vlines([pos[s] for s in got], 0, [gap[s] for s in got], color=col, lw=0.8)
        ax.scatter(
            [pos[s] for s in got],
            [gap[s] for s in got],
            color=col,
            marker=mk,
            s=26,
            zorder=3,
            label=f"{key}, n={len(got)}",
        )
        if got:
            m = c.mean(gap[s] for s in got)
            ax.hlines(m, pos[got[0]] - 0.4, pos[got[-1]] + 0.4, color=col, lw=1.6, ls="--")
            ax.text(pos[got[-1]] + 0.45, m, f"mean {m:+.2f}", color=col, va="center", fontsize=7.5)
            print(f"{grp}: n={len(got)} mean D-C'M {m:+.2f}")
        missing = [s for s in seeds if s not in gap]
        if missing:
            x0, x1 = pos[missing[0]] - 0.4, pos[missing[-1]] + 0.4
            ax.axvspan(x0, x1, color="#dddddd", alpha=0.6)
            ax.text((x0 + x1) / 2, 1.0, "E3 pending", ha="center", fontsize=8)
    for s in allseeds:
        if s in gap:
            ax.text(pos[s], -4.65, f"{steps[s]}", ha="center", fontsize=7, color=c.DGRAY)
            print(
                f"seed {s}: D {d[s]:.2f} C'M {cm[s]:.2f} gap {gap[s]:+.2f} D burst steps {steps[s]}"
            )
    ax.text(-0.9, -4.65, "steps", ha="right", fontsize=7, color=c.DGRAY)
    allv = [gap[s] for s in gap]
    print(
        f"all {len(allv)} seeds mean {c.mean(allv):+.3f}, wins {sum(v > 0 for v in allv)}/{len(allv)}"
    )
    ax.set_xticks([pos[s] for s in allseeds], [str(s) for s in allseeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("D minus C'-matched (points)")
    ax.set_ylim(-5.0, 2.8)
    ax.set_xlim(-1.0, pos[allseeds[-1]] + 1.9)
    ax.legend(loc="upper right", fontsize=7)
    c.save(fig, "fig6_replication.pdf")


if __name__ == "__main__":
    main()

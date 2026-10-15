"""Fig 5: remap-only (R1) final accuracy vs spare fraction (capacity cliff) with D at 15% spares."""

import _common as c
import matplotlib.pyplot as plt

SEEDS = (42, 101, 102)
R1 = {
    2: "resnet9_deploy_r1_stress_rf02",
    5: "resnet9_deploy_r1_stress_rf05",
    10: "resnet9_deploy_r1_stress_rf10",
    15: "resnet9_deploy_r1_stress_rf15",
    30: "resnet9_deploy_r1_stress_rf30",
}
D15 = "resnet9_deploy_d_stress_rf15"


def main() -> None:
    c.setup()
    fig, ax = plt.subplots(figsize=(c.COL + 0.6, 2.8))
    col, mk, ls, _lab = c.ARM["R1"]
    xs, ms = [], []
    for pct, spec in R1.items():
        v = {s: a for s, a in c.arm(spec).items() if s in SEEDS}
        print(f"R1 @{pct}%: n={len(v)} mean={c.mean(v.values()):.2f} seeds {c.fmtv(v)}")
        ax.scatter(
            [pct + c.jitter(j, 3, 0.5) for j in range(len(v))],
            [v[s] for s in sorted(v)],
            s=14,
            color=col,
            marker=mk,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
        xs.append(pct)
        ms.append(c.mean(v.values()))
    ax.plot(xs, ms, color=col, ls=ls, marker=mk, ms=5.5, lw=1.4, label="R1 remap only, mean")
    dv = {s: a for s, a in c.arm(D15).items() if s in SEEDS}
    dc, dm, _, _ = c.ARM["D"]
    ax.scatter(
        [15 + c.jitter(j, 3, 0.5) for j in range(3)],
        [dv[s] for s in sorted(dv)],
        s=14,
        color=dc,
        marker=dm,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    ax.plot(
        [15],
        [c.mean(dv.values())],
        marker="o",
        ms=7,
        color=dc,
        zorder=4,
        label="D at 15% spares, mean",
    )
    print(f"D @15%: n={len(dv)} mean={c.mean(dv.values()):.2f} seeds {c.fmtv(dv)}")
    clean = c.mean(c.arm(R1[30], "baseline_test_acc").values())
    print(f"clean mean (3 seeds) {clean:.2f}")
    ax.axhline(clean, color=c.GRAY, lw=0.8, ls=(0, (1, 2)))
    ax.text(1.5, clean - 5, f"clean chip {clean:.1f}", fontsize=7.5, color=c.DGRAY)
    cum = 1 - 0.9**3
    print(f"cumulative stuck fraction 1-0.9^3 = {cum * 100:.1f}%")
    ax.axvline(cum * 100, color="black", lw=0.8, ls="--")
    ax.annotate(
        "cumulative stuck\nfraction 27.1%\n(1-0.9$^3$)",
        (cum * 100, 40),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
    )
    ax.annotate(
        "R1 at chance",
        (7.5, ms[2]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        color=c.DGRAY,
        fontsize=7.5,
    )
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 100)
    ax.set_xticks([2, 5, 10, 15, 30])
    ax.set_xlabel("Spare fraction (%)")
    ax.set_ylabel("Final test accuracy (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.46))
    c.save(fig, "fig5_spare_cliff.pdf")


if __name__ == "__main__":
    main()

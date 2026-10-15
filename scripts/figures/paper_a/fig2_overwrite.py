"""Fig 2: frozen-reference (C) vs living-reference (C') per-wave accuracy, old burst rate 0.005."""

import _common as c
import matplotlib.pyplot as plt


def main() -> None:
    c.setup()
    specs = {"C": "resnet9_deploy_c", "C'": "resnet9_deploy_c_living"}
    fig, ax = plt.subplots(figsize=(c.COL, 2.5))
    for k, spec in specs.items():
        col, mk, ls, _ = c.ARM[k]
        lab = {"C": "C (frozen reference)", "C'": "C' (living reference)"}[k]
        w = c.waves(spec)
        seeds = sorted(w)
        n_w = len(next(iter(w.values())))
        x = list(range(1, n_w + 1))
        for s in seeds:
            ax.plot(x, w[s], color=col, lw=0.6, alpha=0.55, ls=ls, marker=mk, ms=2.8, mew=0)
        m = [c.mean(w[s][i] for s in seeds) for i in range(n_w)]
        ax.plot(
            x, m, color=col, lw=1.6, ls=ls, marker=mk, ms=5, label=f"{lab}, mean (n={len(seeds)})"
        )
        ax.annotate(
            k,
            (x[-1], m[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            color=col,
            va="center",
            fontsize=8,
        )
        print(
            f"{k}: per-wave mean {[round(v, 2) for v in m]}; per seed { ({s: w[s] for s in seeds}) }"
        )
        print(f"{k}: final per seed {c.fmtv(c.arm(spec))} mean {c.mean(c.arm(spec).values()):.2f}")
    cf, cl = c.arm(specs["C"]), c.arm(specs["C'"])
    d = [cl[s] - cf[s] for s in sorted(cf)]
    print(f"C' minus C final: {[round(v, 2) for v in d]} mean {c.mean(d):+.2f}")
    ax.set_xticks(x)
    ax.set_xlim(0.8, n_w + 0.7)
    ax.set_xlabel("Fault wave")
    ax.set_ylabel("Test accuracy after wave (%)")
    ax.legend(loc="lower left")
    c.save(fig, "fig2_overwrite.pdf")


if __name__ == "__main__":
    main()

"""Closed-loop extension: verdict_metrics.csv for criteria/x_loop_ext.yaml.

One row per (net, regime, sigma, seed). ResNet-9 seeds 42/101/102 at sigma 2% come from the x_loop run
(--base, dense oracle rows = results/_live9 S1 runs); everything else from the extension mirror (--ext).

  python experiments/spectral/x_loop_ext_summary.py --base results/spectral/x_loop_box/results \
      --ext results/spectral/x_loop_ext/results [--live results/_live9]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from x_loop_summary import _ev, _final, _loc

OLD, NEW = (42, 101, 102), (103, 104)


def build(base: Path, ext: Path, live: Path) -> pd.DataFrame:
    rows = []

    def add(net, regime, sigma, seed, d_est, d_or, floor, c_est=None, c_or=None):
        r = {"net": net, "regime": regime, "sigma": sigma, "seed": seed, "d_est": _final(d_est),
             "d_oracle": _final(d_or), "floor": _final(floor), "c_est": _final(c_est),
             "c_oracle": _final(c_or), **_loc(d_est)}  # fmt: skip
        rows.append(r)

    for s in OLD + NEW:
        src = base if s in OLD else ext
        # oracle rows (shared by both sigmas)
        if s in OLD:
            d_or = _ev(live / f"resnet9_real_s1_d_seed{s}")
            c_or = _ev(live / f"resnet9_real_s1_c_living_matched_seed{s}")
            fl = _ev(live / f"resnet9_real_s1_r1_seed{s}")
        else:
            d_or = _ev(ext / f"x_dense_d_xoracle_seed{s}")
            c_or = _ev(ext / f"x_dense_c_living_matched_xoracle_seed{s}")
            fl = _ev(ext / f"x_dense_r1_xoracle_seed{s}")
        sp_or = _ev(src / f"x_sparse_d_xoracle_seed{s}")
        sp_fl = _ev(src / f"x_sparse_r0_xoracle_seed{s}")
        for sig, root in ((0.02, src), (0.04, ext)):
            tag = f"adaptive_s{sig:g}"
            c_est = _ev(root / f"x_dense_c_living_matched_x{tag}_seed{s}") if sig == 0.02 else None
            add(
                "resnet9",
                "dense",
                sig,
                s,
                _ev(root / f"x_dense_d_x{tag}_seed{s}"),
                d_or,
                fl,
                c_est,
                c_or,
            )
            add("resnet9", "sparse", sig, s, _ev(root / f"x_sparse_d_x{tag}_seed{s}"), sp_or, sp_fl)
    for s in OLD + NEW:
        add(
            "vit",
            "dense",
            0.02,
            s,
            _ev(ext / f"x_vit_dense_d_xadaptive_s0.02_seed{s}"),
            _ev(ext / f"x_vit_dense_d_xoracle_seed{s}"),
            _ev(ext / f"x_vit_dense_r1_xoracle_seed{s}"),
        )
        add(
            "vit",
            "sparse",
            0.02,
            s,
            _ev(ext / f"x_vit_sparse_d_xadaptive_s0.02_seed{s}"),
            _ev(ext / f"x_vit_sparse_d_xoracle_seed{s}"),
            _ev(ext / f"x_vit_sparse_r0_xoracle_seed{s}"),
        )
    df = pd.DataFrame(rows)
    df["gap_d"] = df.d_est - df.d_oracle
    df["d_est_minus_c_est"] = df.d_est - df.c_est
    df["d_oracle_minus_c_oracle"] = df.d_oracle - df.c_oracle
    df["closure"] = (df.d_est - df.floor) / (df.d_oracle - df.floor)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/spectral/x_loop_box/results")
    ap.add_argument("--ext", default="results/spectral/x_loop_ext/results")
    ap.add_argument("--live", default="results/_live9")
    a = ap.parse_args()
    df = build(Path(a.base), Path(a.ext), Path(a.live))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["net", "regime", "sigma", "seed", "d_est", "d_oracle", "floor", "gap_d", "closure", "c_est",
            "c_oracle", "d_est_minus_c_est", "recall_vis_min", "reads_ratio", "fallback_frac", "fp_final"]  # fmt: skip
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    print("\ngroup means:")
    print(df.groupby(["net", "regime", "sigma"])[["gap_d", "closure", "reads_ratio", "recall_vis_min"]]
          .agg(["mean", "min", "max"]).round(3).to_string())  # fmt: skip
    out = Path(a.ext) / "verdict_metrics.csv"
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()

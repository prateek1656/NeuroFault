"""Shared style and run loaders for the Paper A figures.

Reuses the canonical-run collector of scripts/analysis/derive_paper_numbers.py through
paper_a_metrics.build_db / resolve, so only canonical (newest complete copy) runs are used.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import paper_a_metrics as pam

FIGS = ROOT / "papers" / "controller" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, AMBER, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, DGRAY = "#8a8a8a", "#444444"
PAGE, COL = 7.16, 3.5

# arm -> (colour, marker, linestyle, label); the same arm looks the same in every figure.
ARM = {
    "D": (BLUE, "o", "-", "D (controller)"),
    "C'M": (ORANGE, "s", "--", "C'-matched"),
    "C'": (ORANGE, "D", "--", "C' (unmatched)"),
    "C": (GRAY, "P", "--", "C (frozen ref.)"),
    "BM": (AQUA, "^", "-.", "B-matched"),
    "B": (AQUA, "v", "-.", "B (unmatched)"),
    "R1": (AMBER, "d", ":", "R1 (remap only)"),
    "R0": (DGRAY, "X", ":", "R0 (none)"),
    "E1L": (PINK, "p", "-", "E1-last (Liu-style)"),
    "E1F": (PINK, "h", "--", "E1-first"),
    "E2": (DGRAY, "*", "-.", "E2 (BN recal only)"),
    "Liu": (PINK, "p", "-", "Liu-style (own)"),
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e3e3e3",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


_DB: dict | None = None


def db() -> dict:
    global _DB
    if _DB is None:
        _DB = pam.build_db()
    return _DB


def arm(spec: str, key: str = "final_test_acc") -> dict[int, float]:
    return pam.acc(pam.resolve(db(), spec), key)


def rows(spec: str) -> dict[int, dict]:
    return pam.resolve(db(), spec)


def waves(spec: str, key: str = "test_acc_after") -> dict[int, list[float]]:
    return {s: [float(e[key]) for e in pam.events_of(r)] for s, r in rows(spec).items()}


# canonical ResNet-9 main-test specs (same strings as paper_a_metrics.SCEN / sec_heldout)
MAIN = {
    "D": "resnet9_deploy_d_eta3e-1+resnet9_final_d",
    "C'M": "resnet9_deploy_c_living_matched_eta5e-1+resnet9_final_c_living_matched",
    "BM": "resnet9_deploy_b_matched_eta5e-1+resnet9_final_b_matched",
    "B": "resnet9_deploy_b_eta5e-1+resnet9_final_b",
    "R1": "resnet9_deploy_r1+resnet9_final_r1",
    "R0": "resnet9_deploy_r0",
    "E1L": "resnet9_e1_liu_last",
    "E1F": "resnet9_e1_liu_first",
    "E2": "resnet9_e2_bn_recal_only",
}


def mean(xs) -> float:
    return st.fmean(list(xs))


def fmtv(d: dict) -> str:
    return " ".join(f"{s}:{v:.2f}" for s, v in sorted(d.items()))


def jitter(i: int, n: int, w: float = 0.22) -> float:
    return 0.0 if n <= 1 else -w + 2 * w * i / (n - 1)


def save(fig, name: str) -> Path:
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {p}")
    return p

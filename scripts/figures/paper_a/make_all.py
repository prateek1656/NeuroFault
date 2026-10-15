"""Render every Paper A figure into papers/controller/figs/. Run: uv run python scripts/figures/paper_a/make_all.py"""

import subprocess
import sys
from pathlib import Path

import fig2_overwrite
import fig3_main
import fig4_lifetime
import fig5_spare_cliff
import fig6_replication
import fig7_tuning

HERE = Path(__file__).resolve().parent
FIGS = HERE.parents[2] / "papers" / "controller" / "figs"


def fig1() -> None:
    src = HERE / "fig1_overview.tex"
    if subprocess.run(
        ["tectonic", "-X", "compile", "--outdir", str(FIGS), str(src)], check=False
    ).returncode:
        sys.exit("tectonic failed for fig1")
    print(f"wrote {FIGS / 'fig1_overview.pdf'}")


def main() -> None:
    fig1()
    for m in (
        fig2_overwrite,
        fig3_main,
        fig4_lifetime,
        fig5_spare_cliff,
        fig6_replication,
        fig7_tuning,
    ):
        print(f"\n== {m.__name__}")
        m.main()


if __name__ == "__main__":
    main()

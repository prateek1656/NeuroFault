"""Thin wrapper: the Paper B figure code lives in scripts/figures/paper_b/make_figs.py (STYLE.md)."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "figures" / "paper_b" / "make_figs.py"), run_name="__main__")

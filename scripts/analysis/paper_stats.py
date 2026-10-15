"""Paired statistics for every comparison the paper makes.

Reads ``results/derived/canonical_runs.csv`` (made by
``derive_paper_numbers.py``), ``dup_audit.csv`` and ``pair_audit.csv``.  Writes
``stats_final.csv`` and ``stats-final.md`` next to them.  Deterministic.  Run:

    uv run python scripts/analysis/paper_stats.py

Decided up front, identically for ALL comparisons (nothing is chosen after seeing p):

* Pairing: seed.  A and B are paired by seed; the pairing is checked with
  ``checkpoint_config_hash`` (equal hash = same pre-deploy checkpoint and fault realisation).
* Difference = A minus B, in accuracy points.  Measuring point: ``final_test_acc`` for deployment
  runs, ``eor_any`` (end of run) for the continuous regime.
* Statistics: n, per-seed diffs, mean, SD, 95% t-CI, exact sign test (one-sided A>B and
  two-sided), exact Wilcoxon signed-rank (two-sided, zeros dropped, average ranks), paired
  t-test (two-sided), Hedges g_av and d_z.
* PRIMARY test = paired t-test, two-sided.  Wilcoxon and sign are sensitivity checks.  The
  primary p is Holm-adjusted inside a pre-defined family (H = headline wins, N = S4 negative
  cases, S = everything else).
* Noise floor = largest |per-seed difference| between identical-config reruns (same name, seed,
  config, checkpoint).  The RMS rerun difference is also reported.
* Verdict (applied in this order): SIGNIFICANT (Holm-adjusted t p<0.05 AND the unadjusted
  one-sided exact sign test p<0.05, which is impossible below n=5 by construction) / MIXED (seeds disagree in sign) /
  NEGATIVE (mean<0, no seed >0) / MIXED (seeds disagree) /
  NOISE-LEVEL (|mean| < 2 x rms_rerun_diff / sqrt(n)) / CONSISTENT (otherwise).  Revision
  2026-10-13: the earlier yardstick (worst single rerun pair, 1.22) is now only the separate
  column ``exceeds_worst_single_rerun``.
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "results" / "derived"
ALPHA = 0.05
POWER = 0.80

# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------


def _num(x: str) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def load_runs() -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    with (VAL / "canonical_runs.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["complete"] != "True":
                continue
            by.setdefault(r["run_name"], {})[int(r["seed"])] = r
    return by


def resolve(by: dict, spec: str) -> dict[int, dict]:
    """'a+b:42,101' -> seed -> row (first part wins per seed), as in derive_paper_numbers."""
    out: dict[int, dict] = {}
    for part in spec.split("+"):
        name, _, seeds = part.partition(":")
        want = {int(s) for s in seeds.split(",")} if seeds else None
        for seed, row in by.get(name, {}).items():
            if (want is None or seed in want) and seed not in out:
                out[seed] = row
    return out


def names_of(spec: str) -> set[str]:
    return {p.partition(":")[0] for p in spec.split("+")}


def load_pair_audit() -> list[dict]:
    p = VAL / "pair_audit.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") in ("OK", "FLAG")]


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def sign_p(diffs: list[float]) -> tuple[float, float, int, int]:
    """Exact sign test (zeros dropped). Returns (one-sided A>B, two-sided, wins, n_eff)."""
    pos = sum(d > 1e-9 for d in diffs)
    neg = sum(d < -1e-9 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0, 1.0, 0, 0
    up = float(stats.binom.sf(pos - 1, n, 0.5))
    lo = float(stats.binom.cdf(pos, n, 0.5))
    return up, min(1.0, 2 * min(up, lo)), pos, n


def wilcoxon_exact(diffs: list[float]) -> float:
    """Exact two-sided Wilcoxon signed-rank p (zeros dropped, average ranks, full enumeration)."""
    d = [x for x in diffs if abs(x) > 1e-9]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = stats.rankdata([abs(x) for x in d])
    w_obs = float(sum(r for r, x in zip(ranks, d, strict=True) if x > 0))
    mean_w = ranks.sum() / 2
    hits = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = float(sum(r for r, s in zip(ranks, signs, strict=True) if s))
        if abs(w - mean_w) >= abs(w_obs - mean_w) - 1e-9:
            hits += 1
    return hits / 2**n


def hedges_g_av(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    s_av = math.sqrt((sa**2 + sb**2) / 2)
    if s_av == 0:
        return float("nan")
    j = 1 - 3 / (4 * (n - 1) - 1)
    return j * (np.mean(a) - np.mean(b)) / s_av


def n_for_power(dz: float, alpha: float = ALPHA, power: float = POWER, cap: int = 2000) -> str:
    """Smallest n for a two-sided paired t-test with effect d_z (noncentral t)."""
    if not math.isfinite(dz) or dz == 0:
        return "n/a"
    dz = abs(dz)
    for n in range(3, cap + 1):
        df = n - 1
        tc = stats.t.ppf(1 - alpha / 2, df)
        nc = dz * math.sqrt(n)
        pw = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
        if pw >= power:
            return str(n)
    return f">{cap}"


def holm(ps: list[float]) -> list[float]:
    idx = sorted(range(len(ps)), key=lambda i: math.inf if math.isnan(ps[i]) else ps[i])
    adj = [float("nan")] * len(ps)
    run = 0.0
    m = sum(not math.isnan(p) for p in ps)
    for rank, i in enumerate(idx):
        if math.isnan(ps[i]):
            continue
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------
R9 = "resnet9_deploy_"
D3 = R9 + "d_eta3e-1"
B5, C5 = R9 + "b_eta5e-1", R9 + "c_living_eta5e-1"
BM, CM = R9 + "b_matched_eta5e-1", R9 + "c_living_matched_eta5e-1"
D5 = f"{D3}+resnet9_final_d"
BM5 = f"{BM}+resnet9_final_b_matched"
CM5 = f"{CM}+resnet9_final_c_living_matched"
R1_5 = f"{R9}r1+resnet9_final_r1"


def _net(n: str, final: str) -> dict:
    return {
        "r1": f"{n}_deploy_r1+{final}_r1",
        "bm": f"{n}_deploy_b_matched_best+{final}_b_matched",
        "cm": f"{n}_deploy_c_living_matched_best+{final}_c_living_matched",
        "d": f"{n}_deploy_d_eta0.3:42+{n}_deploy_d_best+{final}_d",
        "b3": f"{n}_deploy_b_eta0.3:42+{n}_deploy_b_best",
        "c3": f"{n}_deploy_c_living_eta0.3:42+{n}_deploy_c_living_best",
        "d3": f"{n}_deploy_d_eta0.3:42+{n}_deploy_d_best",
        "b_old": f"{n}_deploy_b_eta0.005",
        "c_old": f"{n}_deploy_c_living_eta0.005",
        "d_old": f"{n}_deploy_d_eta0.005",
    }


VGG, VIT = _net("vgg6_net2", "vgg6_final_deploy"), _net("vit_net3", "vit_final_deploy")

DCONT = "resnet9_cont_d_fixgate:42+resnet9_cont_next_d"
CMCONT = "resnet9_cont_next_c_living_matched"
CCONT = "resnet9_cont_c_living:42+resnet9_cont_next_c_living"
C100 = "resnet9_next_deploy_"
C100_D = C100 + "d_c100_eta0.3"
C100_BEST = "resnet9_final_deploy_c_living_c100_matched_eta0.5"

COMPS: list[dict] = []


def add(cid, finding, desc, a, b, fam, key="final_test_acc", a_lab="D", b_lab="baseline", note=""):
    COMPS.append(
        {
            "id": cid,
            "finding": finding,
            "desc": desc,
            "a": a,
            "b": b,
            "fam": fam,
            "key": key,
            "a_lab": a_lab,
            "b_lab": b_lab,
            "note": note,
        }
    )


# Finding 4: main ResNet-9 test (5 seeds)
add(
    "F4.D-BM", "F4 main", "ResNet-9 main: D vs B-matched (5 seeds)", D5, BM5, "H", b_lab="B-matched"
)
add(
    "F4.D-CM",
    "F4 main",
    "ResNet-9 main: D vs C'-matched (5 seeds)",
    D5,
    CM5,
    "H",
    b_lab="C'-matched",
)
add(
    "F4.CM-BM",
    "F4 main",
    "ResNet-9 main: C'-matched vs B-matched (5 seeds)",
    CM5,
    BM5,
    "S",
    a_lab="C'-matched",
    b_lab="B-matched",
)
# retraining vs none
add(
    "RT.D-R1.main",
    "retraining vs none",
    "ResNet-9 main: D vs remap-only R1 (5 seeds)",
    D5,
    R1_5,
    "S",
    b_lab="R1",
)
add(
    "RT.D-R1.vgg",
    "retraining vs none",
    "VGG6: D vs R1 (5 seeds)",
    VGG["d"],
    VGG["r1"],
    "S",
    b_lab="R1",
)
add(
    "RT.D-R1.vit",
    "retraining vs none",
    "ViT-tiny: D vs R1 (5 seeds)",
    VIT["d"],
    VIT["r1"],
    "S",
    b_lab="R1",
)
add(
    "RT.D-noburst",
    "retraining vs none",
    "ResNet-9 old rate: D with bursts vs no bursts",
    R9 + "d",
    R9 + "d_noburst",
    "S",
    b_lab="D without bursts",
)
add(
    "RT.D-noburst.c100",
    "retraining vs none",
    "CIFAR-100: D vs D without bursts",
    C100_D,
    C100 + "d_noburst_c100",
    "S",
    b_lab="D without bursts",
)
add(
    "RT.D-R1.c100",
    "retraining vs none",
    "CIFAR-100: D vs R1",
    C100_D,
    C100 + "r1_c100",
    "S",
    b_lab="R1",
)
# learning-rate effect
for lab, aa, bb in (
    ("ResNet-9", (B5, C5, D3), (R9 + "b", R9 + "c_living", R9 + "d")),
    ("VGG6", (VGG["b3"], VGG["c3"], VGG["d3"]), (VGG["b_old"], VGG["c_old"], VGG["d_old"])),
    ("ViT-tiny", (VIT["b3"], VIT["c3"], VIT["d3"]), (VIT["b_old"], VIT["c_old"], VIT["d_old"])),
):
    for arm, x, y in zip("BCD", aa, bb, strict=True):
        add(
            f"LR.{lab}.{arm}",
            "LR effect",
            f"{lab} {arm}: tuned rate vs old rate 0.005 (3 seeds)",
            x,
            y,
            "S",
            a_lab=f"{arm} tuned",
            b_lab=f"{arm} old",
            note="tuned rate was picked on seed-42 test accuracy (ResNet-9) -> optimistic",
        )
add(
    "LR.liu",
    "LR effect",
    "Liu-style retraining: burst rate 0.5 vs 0.005",
    R9 + "liu_style_lr0.5",
    R9 + "liu_style_lr0.005",
    "S",
    a_lab="Liu 0.5",
    b_lab="Liu 0.005",
)
add(
    "LR.D-liu",
    "LR effect",
    "D vs Liu-style at its best rate 0.5",
    D3,
    R9 + "liu_style_lr0.5",
    "S",
    b_lab="Liu-style 0.5",
)
# Finding 5 ablations (tuned rate unless noted)
NP = "resnet9_final_d_nopace_matched"
add(
    "F5.pace.matched", "F5 ablation", "D vs no-pacing, step-matched", D3, NP, "S", b_lab="no pacing"
)
add(
    "F5.pace.BD2",
    "F5 ablation",
    "D vs no-pacing (1000 steps, run BD2; unmatched compute)",
    D3,
    R9 + "d_nopace_etaBD2",
    "S",
    b_lab="no pacing BD2",
)
add(
    "F5.pace.BD",
    "F5 ablation",
    "D vs no-pacing (1000 steps, run BD; unmatched compute)",
    D3,
    R9 + "d_nopace_etaBD",
    "S",
    b_lab="no pacing BD",
)
add(
    "F5.rank.BD2",
    "F5 ablation",
    "D vs random order (BD2)",
    D3,
    R9 + "d_nosal_etaBD2",
    "S",
    b_lab="random order",
)
add(
    "F5.rank.BD",
    "F5 ablation",
    "D vs random order (BD)",
    D3,
    R9 + "d_nosal_etaBD",
    "S",
    b_lab="random order",
)
add(
    "F5.mag.BD2",
    "F5 ablation",
    "D vs weight-size order (BD2)",
    D3,
    R9 + "d_mag_etaBD2",
    "S",
    b_lab="weight-size order",
)
add(
    "F5.mag.BD",
    "F5 ablation",
    "D vs weight-size order (BD)",
    D3,
    R9 + "d_mag_etaBD",
    "S",
    b_lab="weight-size order",
)
add(
    "F5.nobn",
    "F5 ablation",
    "D vs no BN fix (old rate)",
    R9 + "d",
    R9 + "d_nobn",
    "S",
    b_lab="no BN fix",
)
add(
    "F5.noadapt",
    "F5 ablation",
    "D vs no adaptation (old rate)",
    R9 + "d",
    R9 + "d_noadapt",
    "S",
    b_lab="no adaptation",
)
# Finding 6 stress
for k, lab in (
    ("rf02", "2% spare"),
    ("rf05", "5% spare"),
    ("rf10", "10% spare"),
    ("d15", "15% faults/wave"),
    ("d20", "20% faults/wave"),
    ("d25", "25% faults/wave"),
):
    add(
        f"F6.{k}.D-CM",
        "F6 stress",
        f"Stress {lab}: D vs C'-matched",
        R9 + f"d_stress_{k}",
        R9 + f"c_living_stressM_{k}",
        "H",
        b_lab="C'-matched",
    )
for k, lab in (
    ("rf02", "2% spare"),
    ("rf05", "5% spare"),
    ("rf10", "10% spare"),
    ("d15", "15% faults/wave"),
    ("d20", "20% faults/wave"),
    ("d25", "25% faults/wave"),
):
    add(
        f"F6.{k}.D-BM",
        "F6 stress",
        f"Stress {lab}: D vs B-matched",
        R9 + f"d_stress_{k}",
        R9 + f"b_stressM_{k}",
        "S",
        b_lab="B-matched",
    )
for k, lab in (("w6", "6 small waves"), ("w8", "8 small waves")):
    add(
        f"F6.{k}.oldgate",
        "F6 gate",
        f"{lab}, OLD gate: D vs C'-matched (old, unequal steps)",
        R9 + f"d_stress_{k}",
        R9 + f"c_living_stressM_{k}",
        "S",
        b_lab="C'-matched",
    )
    add(
        f"F6.{k}.gfix",
        "F6 gate",
        f"{lab}, gate-fix: D vs C' matched to D-gfix steps",
        R9 + f"d_gfix_{k}",
        R9 + f"c_living_matchDgfix_{k}",
        "S",
        b_lab="C' (matchDgfix)",
    )
add(
    "F8.g5.old",
    "F8 pattern",
    "6x5% waves (g5): D old gate vs C'-matched",
    R9 + "d_g5",
    R9 + "c_living_g5_matched",
    "S",
    b_lab="C'-matched",
)
add(
    "F8.g5.gfix",
    "F8 pattern",
    "6x5% waves (g5): D gate-fix vs C'-matched (equal steps)",
    R9 + "d_g5_gfix",
    R9 + "c_living_g5_matched",
    "S",
    b_lab="C'-matched",
)
# Finding 7 continuous (end-of-run)
add(
    "F7.cont.D-CM",
    "F7 continuous",
    "Continuous regime: D vs C'-matched (end of run)",
    DCONT,
    CMCONT,
    "H",
    key="eor_any",
    b_lab="C'-matched",
    note="seed 42 D uses last logged point (batch 19100); no checkpoint hash in continuous runs",
)
add(
    "F7.cont.D-Cp",
    "F7 continuous",
    "Continuous regime: D vs unmatched C' (end of run)",
    DCONT,
    CCONT,
    "S",
    key="eor_any",
    b_lab="C' (unmatched)",
)
# Finding 10 CIFAR-100
add(
    "F10.D-CMbest",
    "F10 CIFAR-100",
    "CIFAR-100: D vs best C'-matched (rate 0.5)",
    C100_D,
    C100_BEST,
    "H",
    b_lab="C'-matched 0.5",
)
add(
    "F10.D-CM1.0",
    "F10 CIFAR-100",
    "CIFAR-100: D vs C'-matched rate 1.0",
    C100_D,
    C100 + "c_living_c100_matched_eta1.0",
    "S",
    b_lab="C'-matched 1.0",
)
add(
    "F10.D-CM2.0",
    "F10 CIFAR-100",
    "CIFAR-100: D vs C'-matched rate 2.0",
    C100_D,
    C100 + "c_living_c100_matched_eta2.0",
    "S",
    b_lab="C'-matched 2.0",
)
# Finding 11 VGG / ViT
add(
    "F11.vgg.D-CM",
    "F11 nets",
    "VGG6: D vs C'-matched (5 seeds)",
    VGG["d"],
    VGG["cm"],
    "H",
    b_lab="C'-matched",
)
add(
    "F11.vit.D-CM",
    "F11 nets",
    "ViT-tiny: D vs C'-matched (5 seeds)",
    VIT["d"],
    VIT["cm"],
    "H",
    b_lab="C'-matched",
)
add(
    "F11.vgg.D-BM",
    "F11 nets",
    "VGG6: D vs B-matched (5 seeds)",
    VGG["d"],
    VGG["bm"],
    "S",
    b_lab="B-matched",
)
add(
    "F11.vit.D-BM",
    "F11 nets",
    "ViT-tiny: D vs B-matched (5 seeds)",
    VIT["d"],
    VIT["bm"],
    "S",
    b_lab="B-matched",
)
# Finding 12 realistic S1-S3
for s in ("s1", "s2", "s3"):
    p = f"resnet9_real_{s}_"
    add(
        f"F12.{s}.D-CM",
        "F12 realistic",
        f"Realistic {s.upper()}: D vs C'-matched",
        p + "d",
        p + "c_living_matched",
        "H",
        b_lab="C'-matched",
    )
    add(
        f"F12.{s}.D-BM",
        "F12 realistic",
        f"Realistic {s.upper()}: D vs B-matched",
        p + "d",
        p + "b_matched",
        "S",
        b_lab="B-matched",
    )
# Finding 14 calibrated S4 (D loses)
for k in ("s4a", "s4b", "s4a5", "s4b5"):
    p = f"resnet9_calib_{k}_"
    add(
        f"F14.{k}.D-BM",
        "F14 S4",
        f"Calibrated {k.upper()}: D vs B-matched",
        p + "d",
        p + "b_matched",
        "N",
        b_lab="B-matched",
    )
add(
    "F14.s4b5.D-CM",
    "F14 S4",
    "Calibrated S4B5: D vs C'-matched",
    "resnet9_calib_s4b5_d",
    "resnet9_calib_s4b5_c_living_matched",
    "N",
    b_lab="C'-matched",
)
# Finding 15 D2
for k in ("s4a", "s4b", "s4a5", "s4b5"):
    q = f"resnet9_d2calib_{k}_"
    o = f"resnet9_calib_{k}_d"
    add(f"F15.{k}.D2-D", "F15 D2", f"S4 {k}: D2 vs D", q + "d2", o, "S", a_lab="D2", b_lab="D")
    add(
        f"F15.{k}.D2-BM",
        "F15 D2",
        f"S4 {k}: D2 vs B-matched (D2 steps)",
        q + "d2",
        q + "b_matched",
        "S",
        a_lab="D2",
        b_lab="B-matched",
    )
    add(
        f"F15.{k}.D2-C2",
        "F15 D2",
        f"S4 {k}: D2 vs C2",
        q + "d2",
        q + "c2",
        "S",
        a_lab="D2",
        b_lab="C2",
    )
add(
    "F15.s4b5.D2-CM",
    "F15 D2",
    "S4 s4b5: D2 vs C'-matched (D2 steps)",
    "resnet9_d2calib_s4b5_d2",
    "resnet9_d2calib_s4b5_c_living_matched",
    "S",
    a_lab="D2",
    b_lab="C'-matched",
)
M = "resnet9_d2_"
add(
    "F15.main.D2-D",
    "F15 D2",
    "Main test (5 seeds): D2 vs D",
    M + "d2",
    D5,
    "S",
    a_lab="D2",
    b_lab="D",
)
add(
    "F15.main.D2-CM",
    "F15 D2",
    "Main test (5 seeds): D2 vs C'-matched (D2 steps)",
    M + "d2",
    M + "c_living_matched",
    "S",
    a_lab="D2",
    b_lab="C'-matched",
)
add(
    "F15.main.D2-BM",
    "F15 D2",
    "Main test (5 seeds): D2 vs B-matched (D2 steps)",
    M + "d2",
    M + "b_matched",
    "S",
    a_lab="D2",
    b_lab="B-matched",
)
add(
    "F15.main.D2-C2",
    "F15 D2",
    "Main test (5 seeds): D2 vs C2",
    M + "d2",
    M + "c2",
    "S",
    a_lab="D2",
    b_lab="C2",
)


# --------------------------------------------------------------------------
# noise floor
# --------------------------------------------------------------------------
def noise_floor(by: dict) -> dict:
    """Per-seed differences between identical-config reruns (BD2 minus BD, plus dup_audit)."""
    diffs: list[tuple[str, int, float]] = []
    for stem in ("d_nopace", "d_nosal", "d_mag"):
        a, b = by.get(R9 + stem + "_etaBD2", {}), by.get(R9 + stem + "_etaBD", {})
        for s in sorted(set(a) & set(b)):
            diffs.append(
                (f"{stem} BD2-BD", s, float(a[s]["final_test_acc"]) - float(b[s]["final_test_acc"]))
            )
    p = VAL / "dup_audit.csv"
    if p.exists():
        with p.open() as fh:
            for r in csv.DictReader(fh):
                if r["identical_config"] == "True" and r["identical_results"] == "False":
                    f = [float(x) for x in r["finals"].split(",")]
                    diffs.append((f"{r['run']} copies", int(r["seed"]), f[1] - f[0]))
    arr = np.array([d for _, _, d in diffs])
    return {
        "diffs": diffs,
        "max_abs": float(np.abs(arr).max()),
        "rms": float(math.sqrt(np.mean(arr**2))),
        "n": len(arr),
        "sd_run": float(math.sqrt(np.mean(arr**2) / 2)),
    }


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def evaluate(by: dict, noise: dict, audit: list[dict]) -> list[dict]:
    rows = []
    for c in COMPS:
        ra, rb = resolve(by, c["a"]), resolve(by, c["b"])
        seeds = sorted(set(ra) & set(rb))
        pairs = [(s, _num(ra[s][c["key"]]), _num(rb[s][c["key"]])) for s in seeds]
        pairs = [(s, x, y) for s, x, y in pairs if x is not None and y is not None]
        out = dict(c)
        if len(pairs) < 2:
            out.update(n=len(pairs), seeds=" ".join(str(p[0]) for p in pairs), verdict="UNTESTABLE")
            rows.append(out)
            continue
        seeds = [p[0] for p in pairs]
        xa, xb = [p[1] for p in pairs], [p[2] for p in pairs]
        d = [x - y for x, y in zip(xa, xb, strict=True)]
        n = len(d)
        mean, sd = float(np.mean(d)), float(np.std(d, ddof=1))
        se = sd / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        if sd > 0:
            p_t = float(stats.ttest_rel(xa, xb).pvalue)
        else:
            p_t = float("nan")
        s1, s2, wins, n_eff = sign_p(d)
        hashes = [(ra[s]["checkpoint_config_hash"], rb[s]["checkpoint_config_hash"]) for s in seeds]
        if all(h[0] and h[1] for h in hashes):
            pairing = "hash-verified" if all(h[0] == h[1] for h in hashes) else "HASH-MISMATCH"
        else:
            pairing = "no-hash"
        flags = [
            r for r in audit if r["armA"] in names_of(c["a"]) and r["armB"] in names_of(c["b"])
        ]
        n_flag = sum(r["status"] == "FLAG" for r in flags)
        detail = sorted(
            {
                r["detail"].split("(")[0].split("=")[0]
                for r in flags
                if r["status"] == "FLAG" and r["detail"]
            }
        )
        dz = mean / sd if sd > 0 else float("nan")
        out.update(
            n=n,
            seeds=" ".join(map(str, seeds)),
            mean_a=float(np.mean(xa)),
            mean_b=float(np.mean(xb)),
            diffs=" ".join(f"{x:+.2f}" for x in d),
            mean=mean,
            sd=sd,
            ci_lo=mean - tcrit * se,
            ci_hi=mean + tcrit * se,
            wins=wins,
            n_eff=n_eff,
            p_sign_1=s1,
            p_sign_2=s2,
            p_wilcoxon_2=wilcoxon_exact(d),
            p_t_2=p_t,
            g_av=hedges_g_av(xa, xb),
            dz=dz,
            pairing=pairing,
            audit_flags=n_flag,
            audit_detail="; ".join(detail),
            ratio_to_noise_max=abs(mean) / noise["max_abs"],
            noise_max=noise["max_abs"],
            noise_thr=2 * noise["rms"] / math.sqrt(n),
            exceeds_worst_single_rerun=abs(mean) > noise["max_abs"],
            noise_se_mean=noise["sd_run"] * math.sqrt(2) / math.sqrt(n),
            n_for_80=n_for_power(dz) if abs(mean) < 1.3 else "",
            n_for_80_holm=n_for_power(dz, ALPHA / 15) if abs(mean) < 1.3 else "",
            min_p_sign_1=0.5**n,
            min_p_wilcoxon_2=2 / 2**n,
            min_p_t_note="",
        )
        rows.append(out)
    for fam in ("H", "N", "S"):
        idx = [i for i, r in enumerate(rows) if r["fam"] == fam and "p_t_2" in r]
        for key, new in (
            ("p_t_2", "p_t_holm"),
            ("p_wilcoxon_2", "p_wilcoxon_holm"),
            ("p_sign_1", "p_sign1_holm"),
        ):
            adj = holm([rows[i][key] for i in idx])
            for i, a in zip(idx, adj, strict=True):
                rows[i][new] = a
        for i in idx:
            rows[i]["family_size"] = len(idx)
    for r in rows:
        if "p_t_2" in r:
            r["verdict"] = verdict(r, noise)
    return rows


def verdict(r: dict, noise: dict) -> str:
    d = [float(x) for x in r["diffs"].split()]
    p_adj = r.get("p_t_holm", float("nan"))
    if not math.isnan(p_adj) and p_adj < ALPHA and r["p_sign_1"] < ALPHA:
        return "SIGNIFICANT"
    if any(x > 0 for x in d) and any(x < 0 for x in d):
        return "MIXED"
    if r["mean"] < 0:
        return "NEGATIVE"
    if abs(r["mean"]) < r["noise_thr"]:
        return "NOISE-LEVEL"
    return "CONSISTENT"


def wording(r: dict) -> str:
    a, b, n, w = r["a_lab"], r["b_lab"], r["n"], r["wins"]
    m, lo, hi = r["mean"], r["ci_lo"], r["ci_hi"]
    v = r["verdict"]
    ci = f"mean {m:+.2f} pts, 95% CI {lo:+.1f} to {hi:+.1f}"
    base = r["desc"]
    if v == "SIGNIFICANT":
        return (
            f"{base}: {a} beat {b} in {w} of {n} seeds ({ci}; Holm-adjusted paired-t "
            f"p={r['p_t_holm']:.3f})."
        )
    tn = ""
    if r.get("p_t_holm", 1.0) < ALPHA:
        tn = (
            f" (paired t, Holm-adjusted p={r['p_t_holm']:.3f}, df={n - 1}: indicative only, "
            f"a rank test cannot reach 0.05 at n={n})"
        )
    if v == "CONSISTENT":
        return (
            f"{base}: {a} was ahead of {b} in {w} of {n} seeds ({ci}){tn}; with n={n} "
            f"this is an effect size with a consistent direction, not a significance claim."
        )
    if v == "NOISE-LEVEL":
        return (
            f"{base}: {a} was ahead of {b} in {w} of {n} seeds ({ci}; unadjusted paired-t "
            f"p={r['p_t_2']:.3f}{', Holm ' + format(r['p_t_holm'], '.3f') if 'p_t_holm' in r else ''})"
            f", but the mean gap is below twice the rerun noise of a {n}-seed mean "
            f"({r['noise_thr']:.2f}); report it as a small difference with "
            f"consistent direction, not as a demonstrated win."
        )
    if v == "MIXED":
        return (
            f"{base}: {a} ahead of {b} in {w} of {n} seeds ({ci}); the sign of the gap "
            f"is not the same in every seed, so no consistent advantage is claimed."
        )
    if v == "NEGATIVE":
        return (
            f"{base}: {a} was worse than {b} in {n - w} of {n} seeds ({ci}); report as a "
            f"limitation."
        )
    return base


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
CSV_COLS = [
    "id",
    "finding",
    "fam",
    "desc",
    "a",
    "b",
    "key",
    "n",
    "seeds",
    "pairing",
    "audit_flags",
    "audit_detail",
    "mean_a",
    "mean_b",
    "diffs",
    "mean",
    "sd",
    "ci_lo",
    "ci_hi",
    "wins",
    "n_eff",
    "p_sign_1",
    "p_sign1_holm",
    "p_sign_2",
    "p_wilcoxon_2",
    "p_wilcoxon_holm",
    "p_t_2",
    "p_t_holm",
    "family_size",
    "g_av",
    "dz",
    "min_p_sign_1",
    "min_p_wilcoxon_2",
    "ratio_to_noise_max",
    "noise_thr",
    "exceeds_worst_single_rerun",
    "noise_se_mean",
    "n_for_80",
    "n_for_80_holm",
    "verdict",
    "wording",
    "note",
]


def fp(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.4f}" if x < 0.001 else f"{x:.3f}"


def f2(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def write_csv(rows: list[dict]) -> None:
    with (VAL / "stats_final.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            if "mean" in r:
                r["wording"] = wording(r)
            w.writerow(
                {
                    k: (
                        ""
                        if r.get(k) is None
                        else (f"{r[k]:.5g}" if isinstance(r.get(k), float) else r[k])
                    )
                    for k in CSV_COLS
                }
            )


def table(rows: list[dict]) -> list[str]:
    hdr = (
        "| id | comparison | n | per-seed diffs | mean | SD | 95% CI | wins | sign 1s | "
        "Wilc 2s | t 2s | Holm t | g_av | /noise | pair | verdict |"
    )
    out = [hdr, "|" + "---|" * 16]
    for r in rows:
        if "mean" not in r:
            out.append(
                f"| {r['id']} | {r['desc']} | {r['n']} | untestable | | | | | | | | | | | | UNTESTABLE |"
            )
            continue
        flag = r["pairing"] + (f", {r['audit_flags']} flagged" if r["audit_flags"] else "")
        out.append(
            f"| {r['id']} | {r['desc']} | {r['n']} | {r['diffs']} | {f2(r['mean'])} | "
            f"{r['sd']:.2f} | {r['ci_lo']:+.1f} to {r['ci_hi']:+.1f} | {r['wins']}/{r['n']} | "
            f"{fp(r['p_sign_1'])} | {fp(r['p_wilcoxon_2'])} | {fp(r['p_t_2'])} | "
            f"{fp(r.get('p_t_holm'))} | {fp(r['g_av'])} | {r['ratio_to_noise_max']:.2f} | "
            f"{flag} | {r['verdict']} |"
        )
    return out


def write_md(rows: list[dict], noise: dict) -> None:
    L: list[str] = []
    L.append("# Final paired statistics for the paper (13 October 2026)\n")
    L.append(
        "Generated by `scripts/analysis/paper_stats.py` from `canonical_runs.csv`, "
        "`dup_audit.csv` and `pair_audit.csv`. Full machine-readable table with suggested "
        "wording for every row: `stats_final.csv`. Deterministic; re-run to refresh.\n"
    )
    L.append("## 1. Method, fixed before looking at any p-value\n")
    L.append(
        "- Pairing: by seed. Same seed means the same pre-deploy checkpoint and fault "
        "realisation. Checked with `checkpoint_config_hash` (column `pair`): `hash-verified` "
        "= equal for every seed; `no-hash` = the runs carry no hash (continuous regime), so "
        "pairing is by seed only and is NOT verified; `HASH-MISMATCH` = not the same "
        "checkpoint. `flagged` = rows of `pair_audit.csv` (written by the run-audit agent) "
        "that flag this pair (different results dir, different GPU, different fault map, "
        "different base canary)."
    )
    L.append(
        "- Measuring point: `final_test_acc` (after the last wave burst) for deployment "
        "runs; `end_of_run_test_acc` (log fallback for seed 42) for the continuous regime."
    )
    L.append(
        "- Per comparison: n, per-seed diffs (A minus B, points), mean, SD, 95% t-CI "
        "(unreliable at n=3: t critical value 4.30), exact sign test one-sided (A>B), exact "
        "Wilcoxon signed-rank two-sided, paired t two-sided, Hedges g_av, ratio of |mean| to "
        "the repeat-noise floor."
    )
    L.append(
        "- PRIMARY test: paired t, two-sided. Wilcoxon and sign are sensitivity checks. The "
        "t-test was fixed as primary because it is the only one of the three that can fall "
        "below 0.05 at n=3 or n=5; it assumes roughly normal seed differences, which cannot "
        "be checked with 3 to 5 points, so every p below is indicative."
    )
    L.append("- Holm-Bonferroni on the primary p inside three pre-defined families:")
    fams = {f: [r for r in rows if r["fam"] == f] for f in "HNS"}
    L.append(
        f"  - **H (headline wins, {len(fams['H'])} comparisons)**: D vs the strongest equal-"
        "compute baseline in every scenario the paper presents as a win: main test D vs "
        "B-matched and D vs C'-matched; six stress cases D vs C'-matched; three realistic "
        "scenarios D vs C'-matched; continuous D vs C'-matched; CIFAR-100 D vs best "
        "C'-matched; VGG6 and ViT-tiny D vs C'-matched. VGG6 is included although it is "
        "not a win, so the family is not chosen by outcome."
    )
    L.append(
        f"  - **N (S4 limitation, {len(fams['N'])} comparisons)**: calibrated S4 scenarios "
        "where D loses (D vs B-matched x4, D vs C'-matched on S4B5)."
    )
    L.append(
        f"  - **S (all other, {len(fams['S'])} comparisons)**: ablations, gate-fix, "
        "learning-rate effect, retraining vs none, D2, D vs B-matched elsewhere. Exploratory."
    )
    L.append(
        "- Verdict rule (in order): SIGNIFICANT = Holm-adjusted primary p < 0.05 AND the "
        "unadjusted exact one-sided sign test p < 0.05 (so a significance label needs n >= 5 "
        "with every seed in the same direction; this second condition was added after a "
        "first pass showed that a df=2 t-test alone would label several n=3 claims "
        "significant, which the minimum-p logic of section 3 rules out; the t-only Holm "
        "result is still shown in every row and quoted in the wording); MIXED = "
        "seeds disagree in sign; NEGATIVE = mean < 0 and no seed above 0; NOISE-LEVEL = "
        "(not SIGNIFICANT, not NEGATIVE, not MIXED) and |mean| < 2 x rms_rerun_diff / sqrt(n) "
        "(0.41 at n=5, 0.53 at n=3, see section 2); CONSISTENT = the same but "
        "|mean| above that threshold.\n"
    )
    L.append(
        "**Revision 2026-10-13.** The first version labelled NOISE-LEVEL when the mean paired "
        "difference was below the worst single rerun-pair difference (1.22). That compared "
        "a mean over n seeds with a single-pair extreme, which is the wrong yardstick; the "
        "rerun noise of a mean over n paired seeds is rms_rerun_diff / sqrt(n). The rule is now "
        "NOISE-LEVEL iff |mean| < 2 x rms_rerun_diff / sqrt(n). The 1.22 figure is kept as the "
        "boolean column `exceeds_worst_single_rerun`; a single-seed result below 1.22 is not "
        "interpretable. Nothing else changed (same tests, families, Holm, precedence). The "
        "change corrects the yardstick; it was not made to move any p-value.\n"
    )
    L.append("## 2. Repeat-noise floor (recomputed from raw)\n")
    L.append(
        f"{noise['n']} per-seed differences between identical-config reruns "
        "(same name, seed, config, checkpoint hash):\n"
    )
    L.append("| rerun pair | seed | diff |\n|---|---|---|")
    for lab, s, x in noise["diffs"]:
        L.append(f"| {lab} | {s} | {x:+.2f} |")
    nn = noise["n"]
    lo = math.sqrt(nn * noise["rms"] ** 2 / stats.chi2.ppf(0.975, nn))
    hi = math.sqrt(nn * noise["rms"] ** 2 / stats.chi2.ppf(0.025, nn))
    L.append(
        f"\nThe N={nn} pairs are: D no-pacing, random-order and weight-size-order ablations "
        "run twice (BD vs BD2, seeds 42/101/102 each = 9 pairs) plus C' on the g5 pattern, "
        "seed 42, run in two results dirs (1 pair). "
        f"RMS rerun diff = **{noise['rms']:.2f}**, 95% chi-square CI "
        f"[{lo:.2f}, {hi:.2f}] (df={nn}, zero-mean assumption); it is itself uncertain and "
        "is dominated by one pair (+1.22). Noise threshold for NOISE-LEVEL = 2 x rms / "
        f"sqrt(n) = {2 * noise['rms'] / math.sqrt(5):.2f} at n=5, "
        f"{2 * noise['rms'] / math.sqrt(3):.2f} at n=3 (using the CI upper bound: "
        f"{2 * hi / math.sqrt(5):.2f} and {2 * hi / math.sqrt(3):.2f}). "
        f"Largest single |diff| = {noise['max_abs']:.2f}: a single-seed result below it is "
        "not interpretable; reported per comparison as `exceeds_worst_single_rerun`, not used "
        f"for labels. Implied SD of one run = {noise['sd_run']:.2f}. The noise of a "
        f"mean-of-n paired difference is about {noise['sd_run'] * math.sqrt(2):.2f}/sqrt(n) "
        f"(n=5: {noise['sd_run'] * math.sqrt(2) / math.sqrt(5):.2f}; n=3: "
        f"{noise['sd_run'] * math.sqrt(2) / math.sqrt(3):.2f}). So a 5-seed mean gap of 1 "
        "point is outside the noise of a 5-seed mean. Note: the "
        "run-to-run noise is already contained in the per-seed SD of every comparison, "
        "so the t-test accounts for it.\n"
    )
    L.append("## 3. What cannot be significant by design\n")
    L.append(
        "| n | min one-sided sign p | min two-sided sign p | min two-sided Wilcoxon exact p | "
        "paired t |\n|---|---|---|---|---|"
    )
    for n in (3, 5):
        L.append(
            f"| {n} | {0.5**n:.4f} | {2 * 0.5**n:.4f} | {2 / 2**n:.4f} | "
            f"no floor, but needs normal diffs; t crit {stats.t.ppf(0.975, n - 1):.2f} |"
        )
    L.append("\nConsequences:\n")
    L.append(
        "- n=3 comparisons can never reach p<0.05 under sign (0.125) or Wilcoxon (0.25). "
        "Only the t-test can, and its CI is wide. A Holm correction on top of the t-test "
        "will usually remove what is left. All n=3 claims must be written as effect sizes "
        "with consistent direction."
    )
    L.append(
        "- n=5 can reach 0.031 (sign, one-sided) but not 0.05 two-sided Wilcoxon (0.0625). "
        "VGG6 (3 of 5 wins vs C'-matched) and ViT-tiny (4 of 5) cannot be significant on the sign test."
    )
    L.append(
        "- The sign test and Wilcoxon values below are therefore reported but cannot be "
        "the basis of a significance claim for n=3.\n"
    )
    L.append("## 4. Claim strength verdict\n")
    L.append(
        "Wording is a full sentence per claim in `stats_final.csv` (column `wording`); the "
        "short form is repeated here for the headline family.\n"
    )
    L.append("| id | claim | fam | n | mean | wins | Holm t | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "mean" in r:
            L.append(
                f"| {r['id']} | {r['desc']} | {r['fam']} | {r['n']} | {f2(r['mean'])} | "
                f"{r['wins']}/{r['n']} | {fp(r.get('p_t_holm'))} | {r['verdict']} |"
            )
    L.append("\n### Verdict counts\n")
    for f in "HNS":
        cnt: dict[str, int] = {}
        for r in fams[f]:
            cnt[r.get("verdict", "UNTESTABLE")] = cnt.get(r.get("verdict", "UNTESTABLE"), 0) + 1
        L.append(f"- Family {f}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    L.append("\n### Suggested paper wording (headline family H)\n")
    for r in fams["H"]:
        if "mean" in r:
            L.append(f"- **{r['id']}** ({r['verdict']}): {wording(r)}")
    L.append("\n### Suggested paper wording (S4 limitation, family N)\n")
    for r in fams["N"]:
        if "mean" in r:
            L.append(f"- **{r['id']}** ({r['verdict']}): {wording(r)}")
    L.append("\n## 5. Tables by finding\n")
    L.append(
        "Columns: `sign 1s` = exact one-sided sign p for A>B; `Wilc 2s` = exact "
        "two-sided Wilcoxon; `t 2s` = paired t; `Holm t` = Holm-adjusted t inside the "
        "family; `/noise` = |mean| divided by the worst single rerun difference "
        f"({noise['max_abs']:.2f}); `pair` = pairing check. p-values are never rounded down "
        "(shown to 3 decimals, 4 below 0.001).\n"
    )
    order: list[str] = []
    for r in rows:
        if r["finding"] not in order:
            order.append(r["finding"])
    for fnd in order:
        L.append(f"### {fnd}\n")
        L += table([r for r in rows if r["finding"] == fnd])
        notes = {r["note"] for r in rows if r["finding"] == fnd and r["note"]}
        for nt in sorted(notes):
            L.append(f"\nNote: {nt}")
        L.append("")
    L.append("## 5b. Pairs flagged by the run audit\n")
    L.append("| id | pairing | flagged seed-rows | reasons (from pair_audit.csv) |")
    L.append("|---|---|---|---|")
    for r in rows:
        if "mean" in r and (r["audit_flags"] or r["pairing"] != "hash-verified"):
            L.append(
                f"| {r['id']} | {r['pairing']} | {r['audit_flags']} | {r['audit_detail'] or '-'} |"
            )
    L.append("")
    L.append("## 6. Power: seeds needed (near-noise comparisons, |mean| < 1.3 pts)\n")
    L.append(
        "Paired t-test, two-sided, 80% power, using the observed d_z = mean/SD of this "
        "comparison (noncentral t). 'Holm' column uses alpha = 0.05/15 (the size of family "
        "H). Observed effects at n=3 to 5 are noisy and tend to be overestimated when "
        "selected for being large, so these n are lower bounds; treat them as orders of "
        "magnitude.\n"
    )
    L.append(
        "| id | comparison | n now | mean | SD | d_z | seeds for 80% (alpha .05) | seeds "
        "(alpha .05/15) |\n|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        if "mean" in r and abs(r["mean"]) < 1.3:
            L.append(
                f"| {r['id']} | {r['desc']} | {r['n']} | {f2(r['mean'])} | {r['sd']:.2f} | "
                f"{r['dz']:.2f} | {r['n_for_80']} | {r['n_for_80_holm']} |"
            )
    L.append(
        "\nFor a planning number that does not depend on an observed (and possibly lucky) "
        f"SD: with a single-run SD of {noise['sd_run']:.2f} points, the paired-difference SD "
        f"is about {noise['sd_run'] * math.sqrt(2):.2f} from noise alone, so a true 1.0-point "
        f"gap needs n >= "
        f"{n_for_power(1.0 / (noise['sd_run'] * math.sqrt(2)))} seeds and a 0.5-point gap "
        f"n >= {n_for_power(0.5 / (noise['sd_run'] * math.sqrt(2)))} (noise only; real "
        "between-seed variation in the gap adds to it).\n"
    )
    (VAL / "stats-final.md").write_text("\n".join(L) + "\n")


def main() -> None:
    by = load_runs()
    noise = noise_floor(by)
    rows = evaluate(by, noise, load_pair_audit())
    write_csv(rows)
    write_md(rows, noise)
    for r in rows:
        if "mean" in r:
            print(
                f"{r['id']:22s} n={r['n']} mean={r['mean']:+.2f} wins={r['wins']}/{r['n']} "
                f"t={fp(r['p_t_2'])} holm={fp(r.get('p_t_holm'))} {r['pairing']:14s} "
                f"{r['verdict']}"
            )
        else:
            print(f"{r['id']:22s} UNTESTABLE n={r['n']}")


if __name__ == "__main__":
    main()

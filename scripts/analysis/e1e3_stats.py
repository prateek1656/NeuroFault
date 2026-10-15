"""E1/E2/E3 analysis, exactly as pre-registered (sections 5-7b of the E1-E3 pre-registration).

Reads every run's events.json directly (no cached tables), checks pairing (per-wave fault_map_hash vs the
D run of the same seed), the sanity rules of section 7 (burst counts, step matching), then runs the tests of
stats-final.md: paired t (primary), exact sign, exact Wilcoxon, Holm within family.

  family E (4): E1-D, E1-R1, E2-R1, E2-D
  family H (15): F4.D-CM replaced by the 8-seed D vs C'-matched; the other 14 raw p-values from stats_final.csv
  exploratory: E1-first vs D, E1-first vs E1-last, E3 new seeds only, E3 old seeds only

Writes results/derived/e1e3_stats.csv and e1e3-results.md.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_stats import hedges_g_av, holm, sign_p, wilcoxon_exact

VAL = ROOT / "results/derived"
OLD3, OLD2, NEW3 = [42, 101, 102], [103, 104], [105, 106, 107]
ALL8 = OLD3 + OLD2 + NEW3
RMS = 0.46

R = ROOT / "results"
SRC = {
    "D": {
        **{s: R / f"_live3/resnet9_deploy_d_eta3e-1_seed{s}" for s in OLD3},
        **{s: R / f"_live11/resnet9_final_d_seed{s}" for s in OLD2},
        **{s: R / f"_live18/resnet9_final_d_seed{s}" for s in NEW3},
    },
    "R1": {
        **{s: R / f"_live18/resnet9_final_r1_seed{s}" for s in NEW3},
        **{s: R / f"_live11/resnet9_final_r1_seed{s}" for s in OLD2},
    },
    "CM": {
        **{s: R / f"_live11/resnet9_final_c_living_matched_seed{s}" for s in OLD2},
        **{s: R / f"_live18/resnet9_final_c_living_matched_seed{s}" for s in NEW3},
    },
}
for arm, name in (
    ("E1last", "e1_liu_last"),
    ("E1first", "e1_liu_first"),
    ("E2", "e2_bn_recal_only"),
):
    SRC[arm] = {
        s: R / ("_live19" if s in OLD2 else "_live18") / f"resnet9_{name}_seed{s}" for s in ALL8
    }


def canonical(name: str, seed: int) -> Path:
    with (VAL / "canonical_runs.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["run_name"] == name and int(r["seed"]) == seed and r["complete"] == "True":
                return ROOT / r["source_dir"]
    raise SystemExit(f"no canonical run {name} seed {seed}")


for s in OLD3:
    SRC["R1"][s] = canonical("resnet9_deploy_r1", s)
    SRC["CM"][s] = canonical("resnet9_deploy_c_living_matched_eta5e-1", s)


def load(p: Path) -> dict:
    ev = json.loads((p / "events.json").read_text())
    if not ev.get("complete"):
        raise SystemExit(f"incomplete run {p}")
    return ev


RUNS = {arm: {s: load(p) for s, p in d.items()} for arm, d in SRC.items()}
ACC = {arm: {s: ev["summary"]["final_test_acc"] for s, ev in d.items()} for arm, d in RUNS.items()}


# ---------------------------------------------------------------- checks (section 5 pairing, section 7 sanity)
def hashes(ev: dict) -> list:
    return [w["fault_map_hash"] for w in ev["events"]]


checks: list[str] = []
pair_ok: dict[tuple[str, int], bool] = {}
for arm in ("R1", "CM", "E1last", "E1first", "E2"):
    for s in ALL8:
        ok = hashes(RUNS[arm][s]) == hashes(RUNS["D"][s])
        pair_ok[(arm, s)] = ok
        if not ok:
            checks.append(f"UNPAIRED: {arm} seed {s} fault maps differ from D")
for arm in ("E1last", "E1first"):
    for s in ALL8:
        ev = RUNS[arm][s]
        nb = sum(1 for w in ev["events"] if w.get("burst_batches", 0) > 0)
        steps, dsteps = (
            ev["summary"]["total_burst_batches"],
            RUNS["D"][s]["summary"]["total_burst_batches"],
        )
        if nb != 1:
            checks.append(f"SANITY: {arm} seed {s} has {nb} bursts (want 1)")
        if abs(steps - dsteps) > 1:
            checks.append(f"SANITY: {arm} seed {s} steps {steps} != D {dsteps}")
for s in ALL8:
    if RUNS["E2"][s]["summary"]["total_burst_batches"] != 0:
        checks.append(f"SANITY: E2 seed {s} has nonzero burst")
    if (
        abs(
            RUNS["CM"][s]["summary"]["total_burst_batches"]
            - RUNS["D"][s]["summary"]["total_burst_batches"]
        )
        > 1
    ):
        checks.append(f"SANITY: CM seed {s} steps not matched to D")
    if ACC["E1last"][s] < ACC["R1"][s] + 5:
        checks.append(f"SANITY: E1-last seed {s} below R1+5")
    if ACC["E2"][s] < ACC["R1"][s] - 2:
        checks.append(
            f"SANITY (pre-registered): E2 seed {s} = {ACC['E2'][s]} below R1-2 = {ACC['R1'][s] - 2:.2f}"
        )


# ---------------------------------------------------------------- tests
def compare(cid: str, a: str, b: str, seeds: list[int], fam: str) -> dict:
    seeds = [s for s in seeds if pair_ok.get((a, s), True) and pair_ok.get((b, s), True)]
    xa, xb = [ACC[a][s] for s in seeds], [ACC[b][s] for s in seeds]
    d = [round(x - y, 4) for x, y in zip(xa, xb, strict=True)]
    n = len(d)
    mean = sum(d) / n
    sd = stats.tstd(d) if n > 1 else float("nan")
    p_t = float(stats.ttest_1samp(d, 0).pvalue) if n > 1 and sd > 0 else float("nan")
    p1, p2, wins, neff = sign_p(d)
    return {
        "id": cid,
        "fam": fam,
        "a": a,
        "b": b,
        "n": n,
        "seeds": " ".join(map(str, seeds)),
        "diffs": " ".join(f"{x:+.2f}" for x in d),
        "mean_a": sum(xa) / n,
        "mean_b": sum(xb) / n,
        "mean": mean,
        "sd": sd,
        "wins": wins,
        "n_eff": neff,
        "p_t_2": p_t,
        "p_sign_1": p1,
        "p_sign_2": p2,
        "p_wilcoxon_2": wilcoxon_exact(d),
        "g_av": hedges_g_av(xa, xb),
        "noise_thr": 2 * RMS / math.sqrt(n),
        "d_list": d,
    }


rows = [
    compare("E1-D", "E1last", "D", ALL8, "E"),
    compare("E1-R1", "E1last", "R1", ALL8, "E"),
    compare("E2-R1", "E2", "R1", ALL8, "E"),
    compare("E2-D", "E2", "D", ALL8, "E"),
]
for r, p in zip(rows, holm([r["p_t_2"] for r in rows]), strict=True):
    r["p_holm"], r["family_size"] = p, 4

e3 = compare("F4.D-CM(8)", "D", "CM", ALL8, "H")
with (VAL / "stats_final.csv").open() as fh:
    fam_h = [r for r in csv.DictReader(fh) if r["fam"] == "H"]
assert len(fam_h) == 15 and sum(r["id"] == "F4.D-CM" for r in fam_h) == 1
ps = [e3["p_t_2"] if r["id"] == "F4.D-CM" else float(r["p_t_2"]) for r in fam_h]
adj = holm(ps)
e3["p_holm"] = adj[[r["id"] for r in fam_h].index("F4.D-CM")]
e3["family_size"] = 15
holm_h_new = {r["id"]: (float(r["p_t_holm"]), a) for r, a in zip(fam_h, adj, strict=True)}
rows.append(e3)

for cid, a, b, seeds in (
    ("E3.new", "D", "CM", NEW3),
    ("E3.old", "D", "CM", OLD3 + OLD2),
    ("E1first-D", "E1first", "D", ALL8),
    ("E1first-E1last", "E1first", "E1last", ALL8),
):
    r = compare(cid, a, b, seeds, "S")
    r["p_holm"], r["family_size"] = float("nan"), 1
    rows.append(r)


def verdict(r: dict) -> str:
    d = r["d_list"]
    if r["fam"] in ("E", "H") and r["p_holm"] < 0.05 and r["p_sign_1"] < 0.05:
        return "SIGNIFICANT"
    if any(x > 0 for x in d) and any(x < 0 for x in d):
        return "MIXED"
    if r["mean"] < 0 and not any(x > 0 for x in d):
        return "NEGATIVE"
    if abs(r["mean"]) < r["noise_thr"]:
        return "NOISE-LEVEL"
    return "CONSISTENT"


for r in rows:
    r["verdict"] = verdict(r)
by = {r["id"]: r for r in rows}

# ---------------------------------------------------------------- decision rules (section 7, 7b)
e3n = by["E3.new"]
if e3["mean"] < 0.5 or e3["wins"] <= 5:
    dec_e3 = "FAILURE"
elif e3["wins"] >= 7 and e3["p_holm"] < 0.05 and e3n["wins"] == 3:
    dec_e3 = "SUCCESS"
else:
    dec_e3 = "HOLD"
e1 = by["E1-D"]
if e1["verdict"] in ("CONSISTENT", "SIGNIFICANT") and e1["mean"] > 0:
    dec_e1 = "LIU AHEAD"
elif abs(e1["mean"]) < e1["noise_thr"] or e1["verdict"] == "MIXED":
    dec_e1 = "NOT DISTINGUISHABLE"
elif e1["mean"] <= -0.5 and max(e1["d_list"]) <= 0.5:
    dec_e1 = "D CONFIRMED AHEAD" + (
        "" if e1["verdict"] == "SIGNIFICANT" else " (not SIGNIFICANT in family E)"
    )
else:
    dec_e1 = "NONE OF THE PRE-REGISTERED OUTCOMES"
e2d = by["E2-D"]
dec_e2 = (
    "RETRAINING NEEDED"
    if e2d["mean"] <= -5
    else "RETRAINING NOT NEEDED"
    if e2d["mean"] >= -2
    else "BETWEEN RULES"
)

# ---------------------------------------------------------------- output
cols = [
    "id",
    "fam",
    "a",
    "b",
    "n",
    "seeds",
    "diffs",
    "mean_a",
    "mean_b",
    "mean",
    "sd",
    "wins",
    "n_eff",
    "p_t_2",
    "p_holm",
    "family_size",
    "p_sign_1",
    "p_sign_2",
    "p_wilcoxon_2",
    "g_av",
    "noise_thr",
    "verdict",
]
with (VAL / "e1e3_stats.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


def f(x: float, k: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{k}f}"


arms = ["D", "CM", "E1last", "E1first", "E2", "R1"]
md = [
    "# E1 / E2 / E3 results (counted runs 2026-10-14)",
    "",
    (
        "Generated by `scripts/analysis/e1e3_stats.py` from raw `events.json`. Rules: "
        "the E1-E3 pre-registration (registered before any counted run)."
    ),
    "",
    "## Decisions (pre-registered rules, section 7 / 7b)",
    "",
    (
        f"- **E3 (D vs C'-matched, 8 seeds): {dec_e3}.** Mean {e3['mean']:+.2f}, {e3['wins']}/{e3['n_eff']} seeds ahead, "
        f"Holm p (family H, 15) = {f(e3['p_holm'], 3)}. New seeds 105-107 alone: mean {e3n['mean']:+.2f}, "
        f"{e3n['wins']}/3 ahead (7b rule needs 3/3). Old 5 seeds: mean {by['E3.old']['mean']:+.2f}."
    ),
    (
        f"- **E1 (Liu one-shot, last wave, vs D): {dec_e1}.** Mean E1-D {e1['mean']:+.2f}, max per-seed {max(e1['d_list']):+.2f}, "
        f"D ahead in {sum(x < 0 for x in e1['d_list'])} of {e1['n']} seeds, label {e1['verdict']}."
    ),
    (
        f"- **E2 (BN recal only, vs D): {dec_e2}.** Mean E2-D {e2d['mean']:+.2f}; E2-R1 {by['E2-R1']['mean']:+.2f} "
        f"({by['E2-R1']['verdict']})."
    ),
    "",
    "## Checks",
    "",
]
md += [f"- {c}" for c in checks] or ["- none failed"]
md += [
    "- pairing: per-wave fault_map_hash identical to D for every (arm, seed) not listed above",
    "- E1: exactly one burst per run and steps = D (+-1) for every run not listed above; E2: zero bursts",
    "",
    "## Per-seed final_test_acc",
    "",
    "| seed | " + " | ".join(arms) + " |",
    "|---" * (len(arms) + 1) + "|",
]
md += [f"| {s} | " + " | ".join(f(ACC[a][s]) for a in arms) + " |" for s in ALL8]
md += [
    "",
    "## Comparisons",
    "",
    "| id | fam | A - B | n | per-seed diffs | mean | wins | p_t | p_holm | p_sign1 | p_wilc | g_av | noise thr | verdict |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
md += [
    f"| {r['id']} | {r['fam']} | {r['a']} - {r['b']} | {r['n']} | {r['diffs']} | {r['mean']:+.2f} | "
    f"{r['wins']}/{r['n_eff']} | {f(r['p_t_2'], 4)} | {f(r['p_holm'], 4)} | {f(r['p_sign_1'], 4)} | "
    f"{f(r['p_wilcoxon_2'], 4)} | {f(r['g_av'])} | {r['noise_thr']:.2f} | {r['verdict']} |"
    for r in rows
]
md += [
    "",
    "## Family H after replacing F4.D-CM (Holm over 15, old adj -> new adj)",
    "",
    "| id | Holm p before | Holm p after |",
    "|---|---|---|",
]
md += [f"| {k} | {f(o, 4)} | {f(n, 4)} |" for k, (o, n) in holm_h_new.items()]
(VAL / "e1e3-results.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

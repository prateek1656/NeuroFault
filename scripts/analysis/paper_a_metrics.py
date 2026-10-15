"""Every number used in the Paper A text, derived from raw run files.

Reads ``results/**/events.json`` (and ``results/spectral/x_loop*`` for the closed loop) through
the run collector of ``derive_paper_numbers.py`` (pinned canonical copy of a duplicate, see canonical_pins.csv).
Writes nothing; prints to stdout.  Run::

    uv run python scripts/analysis/paper_a_metrics.py

Decided up front (before looking at any lifetime result):

* Wave accuracy = ``test_acc_after`` of each wave event (accuracy after that wave's burst, i.e.
  the level the chip serves until the next wave).  Pre-burst dip = ``test_acc_before``.
* Floors: 80 %, 85 %, clean-5 pt, clean-2 pt (clean = the run's ``baseline_test_acc``).
* Waves survived at floor F = number of leading waves whose accuracy is >= F (stops at the first
  miss), so it is censored at the number of waves K.
* AUC = mean post-burst accuracy over the K waves (rectangle rule; equals ``lifetime_mean``).
* Paired statistics by seed: exact one-sided sign test, paired t (two-sided), Holm inside the
  lifetime family printed below.  Label rules follow stats-final.md.
* Canary pick rules: seed-42 full sweep, highest canary mean over waves (rule M) or canary after
  the last wave (rule L); 3-seed picks are restricted to the rates that have 3 seeds.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive_paper_numbers as dpn

ROOT = HERE.parents[1]
RES = ROOT / "results"
FLOORS = {"F80": 80.0, "F85": 85.0}
RERUN_RMS = 0.46  # stats-final.md section 2, RMS of 10 identical-config rerun differences


# ------------------------------------------------------------------ data access
def build_db() -> dict[str, dict[int, dict]]:
    canon, _ = dpn.collect()
    by: dict[str, dict[int, dict]] = {}
    for r in canon:
        if r.get("complete"):
            by.setdefault(r["run_name"], {})[int(r["seed"])] = r
    return by


def resolve(by: dict, spec: str) -> dict[int, dict]:
    """'a+b:42,101' -> seed -> row; first part wins per seed."""
    out: dict[int, dict] = {}
    for part in spec.split("+"):
        name, _, seeds = part.partition(":")
        want = {int(s) for s in seeds.split(",")} if seeds else None
        for seed, row in by.get(name, {}).items():
            if (want is None or seed in want) and seed not in out:
                out[seed] = row
    return out


def events_of(row: dict) -> list[dict]:
    p = ROOT / row["source_dir"] / "events.json"
    return json.loads(p.read_text()).get("events") or []


def acc(rows: dict[int, dict], key: str = "final_test_acc") -> dict[int, float]:
    return {s: float(r[key]) for s, r in rows.items() if r.get(key) not in (None, "")}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------ statistics
def paired(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    out = {"n": n, "seeds": seeds, "diffs": d, "mean": mean(d)}
    if n < 2:
        return out | {
            "sd": float("nan"),
            "wins": sum(x > 0 for x in d),
            "p_t": 1.0,
            "p_sign": 1.0,
            "ci": (float("nan"), float("nan")),
        }
    sd = st.stdev(d)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    wins = sum(x > 0 for x in d)
    p_t = 1.0 if sd == 0 and out["mean"] == 0 else float(stats.ttest_1samp(d, 0.0).pvalue)
    if math.isnan(p_t):
        p_t = 0.0
    p_sign = float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)
    return out | {
        "sd": sd,
        "wins": wins,
        "p_t": p_t,
        "p_sign": p_sign,
        "ci": (out["mean"] - tcrit * se, out["mean"] + tcrit * se),
    }


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def verdict(r: dict, p_holm: float) -> str:
    d, n, m = r["diffs"], r["n"], r["mean"]
    if n and any(x > 0 for x in d) and any(x < 0 for x in d):
        return "MIXED"
    if m < 0 and not any(x > 0 for x in d):
        return "NEGATIVE"
    if n >= 5 and r["wins"] == n and p_holm < 0.05 and r["p_sign"] < 0.05:
        return "SIGNIFICANT"
    if abs(m) < 2 * RERUN_RMS / math.sqrt(max(n, 1)):
        return "NOISE-LEVEL"
    return "CONSISTENT"


def fmt(x: float, nd: int = 2) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def show(label: str, r: dict, p_holm: float | None = None) -> None:
    lo, hi = r["ci"]
    v = verdict(r, p_holm if p_holm is not None else r["p_t"])
    if p_holm is None:
        v += " (unadjusted p; use stats-final.md for the Holm label)"
    ph = "" if p_holm is None else f" holm={p_holm:.4f}"
    print(
        f"  {label}: n={r['n']} mean={fmt(r['mean'])} CI[{fmt(lo, 1)},{fmt(hi, 1)}] "
        f"wins={r['wins']}/{r['n']} p_t={r['p_t']:.4f} p_sign1={r['p_sign']:.4f}{ph} -> {v}  "
        f"diffs={' '.join(fmt(x) for x in r['diffs'])}"
    )


def sec_overwrite_and_nets(by: dict) -> None:
    print(
        "\n=== 1. Remap overwrites learning: living (C') vs frozen (C) reference, old rate 0.005 ==="
    )
    c_frozen = acc(resolve(by, "resnet9_deploy_c"))
    c_living = acc(resolve(by, "resnet9_deploy_c_living"))
    show(
        "C' minus C (3 seeds, old rate; not re-measured at the tuned rate)",
        paired(c_living, c_frozen),
    )
    print(
        f"  means: C frozen {mean(c_frozen.values()):.2f}, C' living {mean(c_living.values()):.2f}"
    )
    print("\n=== 1b. Other networks, 5 seeds, tuned rate (D vs matched baselines) ===")
    for net, final in (("vgg6_net2", "vgg6_final_deploy"), ("vit_net3", "vit_final_deploy")):
        d = acc(resolve(by, f"{net}_deploy_d_eta0.3:42+{net}_deploy_d_best+{final}_d"))
        cm = acc(
            resolve(
                by,
                f"{net}_deploy_c_living_matched_best+{final}_c_living_matched",
            )
        )
        bm = acc(resolve(by, f"{net}_deploy_b_matched_best+{final}_b_matched"))
        show(f"{net} D minus C'-matched", paired(d, cm))
        show(f"{net} D minus B-matched", paired(d, bm))


# ------------------------------------------------------------------ section 2: tuning
def sec_tuning(by: dict) -> None:
    print("\n=== 2a. Old burst rate (0.005) vs tuned rate, final test acc, paired by seed ===")
    pairs = {
        "ResNet-9 B": ("resnet9_deploy_b_eta5e-1", "resnet9_deploy_b"),
        "ResNet-9 C'": ("resnet9_deploy_c_living_eta5e-1", "resnet9_deploy_c_living"),
        "ResNet-9 D": ("resnet9_deploy_d_eta3e-1", "resnet9_deploy_d"),
        "ResNet-9 Liu-style": (
            "resnet9_deploy_liu_style_lr0.5",
            "resnet9_deploy_liu_style_lr0.005",
        ),
        "VGG6 B": (
            "vgg6_net2_deploy_b_eta0.3:42+vgg6_net2_deploy_b_best",
            "vgg6_net2_deploy_b_eta0.005",
        ),
        "VGG6 C'": (
            "vgg6_net2_deploy_c_living_eta0.3:42+vgg6_net2_deploy_c_living_best",
            "vgg6_net2_deploy_c_living_eta0.005",
        ),
        "VGG6 D": (
            "vgg6_net2_deploy_d_eta0.3:42+vgg6_net2_deploy_d_best",
            "vgg6_net2_deploy_d_eta0.005",
        ),
        "ViT B": (
            "vit_net3_deploy_b_eta0.3:42+vit_net3_deploy_b_best",
            "vit_net3_deploy_b_eta0.005",
        ),
        "ViT C'": (
            "vit_net3_deploy_c_living_eta0.3:42+vit_net3_deploy_c_living_best",
            "vit_net3_deploy_c_living_eta0.005",
        ),
        "ViT D": (
            "vit_net3_deploy_d_eta0.3:42+vit_net3_deploy_d_best",
            "vit_net3_deploy_d_eta0.005",
        ),
    }
    for lab, (t, o) in pairs.items():
        at, ao = acc(resolve(by, t)), acc(resolve(by, o))
        r = paired(at, ao)
        print(
            f"  {lab}: old={mean(ao.values()):.2f} tuned={mean(at.values()):.2f} "
            f"gain={fmt(r['mean'])} wins={r['wins']}/{r['n']} diffs={' '.join(fmt(x) for x in r['diffs'])}"
        )
    print("  (ResNet-9 tuned rates B 0.5, C' 0.5, D 0.3 were picked on seed-42 TEST accuracy.)")

    print("\n=== 2b. How the D-vs-baseline gap shrinks (ResNet-9, 3 seeds 42/101/102) ===")
    d_old = acc(resolve(by, "resnet9_deploy_d"))
    c_old = acc(resolve(by, "resnet9_deploy_c_living"))
    b_old = acc(resolve(by, "resnet9_deploy_b"))
    d_t = acc(resolve(by, "resnet9_deploy_d_eta3e-1"))
    c_t = acc(resolve(by, "resnet9_deploy_c_living_eta5e-1"))
    b_t = acc(resolve(by, "resnet9_deploy_b_eta5e-1"))
    seeds3 = (42, 101, 102)
    cm = acc(resolve(by, "resnet9_deploy_c_living_matched_eta5e-1"))
    bm = acc(resolve(by, "resnet9_deploy_b_matched_eta5e-1"))
    for lab, a, b in (
        ("old rate, unmatched D-C'", d_old, c_old),
        ("old rate, unmatched D-B", d_old, b_old),
        ("tuned, unmatched D-C'", d_t, c_t),
        ("tuned, unmatched D-B", d_t, b_t),
        ("tuned, matched D-C'M (3 seeds)", d_t, cm),
        ("tuned, matched D-BM (3 seeds)", d_t, bm),
    ):
        sub = {s: v for s, v in a.items() if s in seeds3}
        r = paired(sub, {s: v for s, v in b.items() if s in seeds3})
        print(f"  {lab}: mean gap {fmt(r['mean'])} wins {r['wins']}/{r['n']}")
    steps = lambda spec: mean(r["total_optimizer_steps"] for r in resolve(by, spec).values())
    print(
        f"  mean burst steps: D {steps('resnet9_deploy_d_eta3e-1'):.0f}, C' unmatched "
        f"{steps('resnet9_deploy_c_living_eta5e-1'):.0f}, B unmatched {steps('resnet9_deploy_b_eta5e-1'):.0f}, "
        f"C'M {steps('resnet9_deploy_c_living_matched_eta5e-1'):.0f}"
    )

    print("\n=== 2c. CIFAR-100: C'-matched under each burst rate (3 seeds), D rate 0.3 ===")
    d100 = acc(resolve(by, "resnet9_next_deploy_d_c100_eta0.3"))
    c100 = {
        "0.3": "resnet9_final_deploy_c_living_c100_matched_eta0.3",
        "0.5": "resnet9_final_deploy_c_living_c100_matched_eta0.5",
        "1.0": "resnet9_next_deploy_c_living_c100_matched_eta1.0",
        "2.0": "resnet9_next_deploy_c_living_c100_matched_eta2.0",
    }
    canary_mean_c100: dict[str, float] = {}
    test_c100: dict[str, float] = {}
    for lr, nm in c100.items():
        rows = resolve(by, nm)
        a = acc(rows)
        test_c100[lr] = mean(a.values())
        canary_mean_c100[lr] = mean(r["canary_mean"] for r in rows.values())
        r = paired(d100, a)
        print(
            f"  C'M rate {lr}: test {test_c100[lr]:.2f} canary_mean {canary_mean_c100[lr]:.2f} "
            f"D minus it {fmt(r['mean'])} ({r['wins']}/{r['n']})"
        )
    pick_t, pick_c = (
        max(test_c100, key=test_c100.get),
        max(canary_mean_c100, key=canary_mean_c100.get),
    )
    print(
        f"  test-picked rate {pick_t}: gap {fmt(mean(d100.values()) - test_c100[pick_t])}; "
        f"canary-picked rate {pick_c}: gap {fmt(mean(d100.values()) - test_c100[pick_c])} "
        f"(D itself at 0.3 only; its rate was test-picked on seed 42)"
    )

    print("\n=== 2d. Compute matching changed a conclusion ===")
    d = acc(resolve(by, "resnet9_deploy_d_eta3e-1"))
    for lab, spec in (
        ("no-pacing 1000 steps (BD2)", "resnet9_deploy_d_nopace_etaBD2"),
        ("no-pacing step-matched", "resnet9_final_d_nopace_matched"),
    ):
        a = acc(resolve(by, spec))
        steps_a = mean(r["total_optimizer_steps"] for r in resolve(by, spec).values())
        r = paired({s: v for s, v in d.items() if s in a}, a)
        print(
            f"  pacing ablation {lab}: D minus it {fmt(r['mean'])} ({r['wins']}/{r['n']}), "
            f"ablated steps {steps_a:.0f}"
        )
    for w in ("w6", "w8"):
        dg = acc(resolve(by, f"resnet9_deploy_d_gfix_{w}"))
        for lab, spec in (
            ("old C' (fewer steps)", f"resnet9_deploy_c_living_stressM_{w}"),
            ("C' matched to gate-fixed D steps", f"resnet9_deploy_c_living_matchDgfix_{w}"),
        ):
            rows = resolve(by, spec)
            r = paired(dg, acc(rows))
            print(
                f"  {w} gate-fixed D vs {lab}: gap {fmt(r['mean'])} ({r['wins']}/{r['n']}), "
                f"steps D {mean(x['total_optimizer_steps'] for x in resolve(by, f'resnet9_deploy_d_gfix_{w}').values()):.0f}"
                f" vs {mean(x['total_optimizer_steps'] for x in rows.values()):.0f}"
            )


def sec_canary(by: dict) -> None:
    print("\n=== 2e. Canary-picked vs test-picked burst rate (ResNet-9, unmatched sweeps) ===")
    arms = {
        "B": "resnet9_deploy_b_eta",
        "C'": "resnet9_deploy_c_living_eta",
        "D": "resnet9_deploy_d_eta",
    }
    table: dict[str, dict[float, dict]] = {}
    for arm, stem in arms.items():
        table[arm] = {}
        for lr, suf in dpn.LR_NAMES.items():
            rows = resolve(by, stem + suf)
            if not rows:
                continue
            table[arm][float(lr)] = {
                "seeds": sorted(rows),
                "test42": rows[42]["final_test_acc"] if 42 in rows else None,
                "last42": rows[42]["canary_last"] if 42 in rows else None,
                "mean42": rows[42]["canary_mean"] if 42 in rows else None,
                "test3": mean(r["final_test_acc"] for r in rows.values())
                if len(rows) >= 3
                else None,
                "last3": mean(r["canary_last"] for r in rows.values()) if len(rows) >= 3 else None,
                "mean3": mean(r["canary_mean"] for r in rows.values()) if len(rows) >= 3 else None,
            }
    picks: dict[tuple[str, str], float] = {}
    for arm, t in table.items():
        best42 = max(t, key=lambda k: t[k]["test42"])
        pm = max(t, key=lambda k: t[k]["mean42"])
        pl = max(t, key=lambda k: t[k]["last42"])
        t3 = {k: v for k, v in t.items() if v["test3"] is not None}
        best3 = max(t3, key=lambda k: t3[k]["test3"])
        pm3 = max(t3, key=lambda k: t3[k]["mean3"])
        pl3 = max(t3, key=lambda k: t3[k]["last3"])
        print(
            f"  {arm} seed 42 sweep ({len(t)} rates): test-pick {best42} ({t[best42]['test42']:.2f}); "
            f"rule M pick {pm} ({t[pm]['test42']:.2f}, lost {t[best42]['test42'] - t[pm]['test42']:.2f}); "
            f"rule L pick {pl} ({t[pl]['test42']:.2f}, lost {t[best42]['test42'] - t[pl]['test42']:.2f})"
        )
        print(
            f"  {arm} 3-seed set {sorted(t3)}: test-pick {best3} ({t3[best3]['test3']:.2f}); "
            f"rule M pick {pm3} ({t3[pm3]['test3']:.2f}, lost {t3[best3]['test3'] - t3[pm3]['test3']:.2f}); "
            f"rule L pick {pl3} ({t3[pl3]['test3']:.2f}, lost {t3[best3]['test3'] - t3[pl3]['test3']:.2f})"
        )
        picks[(arm, "test3")], picks[(arm, "M3")], picks[(arm, "L3")] = best3, pm3, pl3
        picks[(arm, "test42")], picks[(arm, "M42")], picks[(arm, "L42")] = best42, pm, pl

    print("  Headline movement, unmatched arms (D minus B, D minus C'), mean over rates' 3 seeds:")
    for rule in ("test3", "M3", "L3"):
        v = {a: table[a][picks[(a, rule)]]["test3"] for a in arms}
        cp = "C'"
        print(
            f"    {rule}: rates B {picks[('B', rule)]}, C' {picks[(cp, rule)]}, D {picks[('D', rule)]}; "
            f"B {v['B']:.2f} C' {v[cp]:.2f} D {v['D']:.2f}; D-B {v['D'] - v['B']:+.2f}, "
            f"D-C' {v['D'] - v[cp]:+.2f}"
        )
    base_b = table["B"][picks[("B", "test3")]]["test3"]
    base_c = table["C'"][picks[("C'", "test3")]]["test3"]
    for rule in ("M3", "L3"):
        db = table["B"][picks[("B", rule)]]["test3"] - base_b
        dc = table["C'"][picks[("C'", rule)]]["test3"] - base_c
        print(
            f"    projected shift of the matched baselines under {rule} (change in the unmatched "
            f"arm's own accuracy; matched runs exist only at 0.5): B {db:+.2f}, C' {dc:+.2f}"
        )
    print(
        "  (Matched B/C' arms exist only at rate 0.5, so canary-picked matched numbers are projections.)"
    )


def sec_heldout(by: dict) -> None:
    print(
        "\n=== 2f. Main-test margins without seed 42 (rates were picked on seed-42 test accuracy) ==="
    )
    d = acc(resolve(by, "resnet9_deploy_d_eta3e-1+resnet9_final_d"))
    for lab, spec in (
        ("B-matched", "resnet9_deploy_b_matched_eta5e-1+resnet9_final_b_matched"),
        ("C'-matched", "resnet9_deploy_c_living_matched_eta5e-1+resnet9_final_c_living_matched"),
    ):
        o = acc(resolve(by, spec))
        for tag, keep in (("all 5 seeds", None), ("seeds 101-104 only", {101, 102, 103, 104})):
            r = paired(
                {s: v for s, v in d.items() if keep is None or s in keep},
                {s: v for s, v in o.items() if keep is None or s in keep},
            )
            print(
                f"  D minus {lab}, {tag}: n={r['n']} mean {fmt(r['mean'])} CI[{fmt(r['ci'][0], 1)},"
                f"{fmt(r['ci'][1], 1)}] wins {r['wins']}/{r['n']} diffs "
                f"{' '.join(fmt(x) for x in r['diffs'])}"
            )


# ------------------------------------------------------------------ section 3: spares
def sec_spares(by: dict) -> None:
    print("\n=== 3. Spare-area sweep (ResNet-9, 3 waves x 10%, 3 seeds 42/101/102) ===")
    sweep = {
        2: (
            "resnet9_deploy_d_stress_rf02",
            "resnet9_deploy_c_living_stressM_rf02",
            "resnet9_deploy_b_stressM_rf02",
            "resnet9_deploy_r1_stress_rf02",
        ),
        5: (
            "resnet9_deploy_d_stress_rf05",
            "resnet9_deploy_c_living_stressM_rf05",
            "resnet9_deploy_b_stressM_rf05",
            "resnet9_deploy_r1_stress_rf05",
        ),
        10: (
            "resnet9_deploy_d_stress_rf10",
            "resnet9_deploy_c_living_stressM_rf10",
            "resnet9_deploy_b_stressM_rf10",
            "resnet9_deploy_r1_stress_rf10",
        ),
        15: (
            "resnet9_deploy_d_stress_rf15",
            "resnet9_deploy_c_living_matched_eta5e-1",
            "resnet9_deploy_b_matched_eta5e-1",
            "resnet9_deploy_r1_stress_rf15",
        ),
        30: (
            "resnet9_deploy_d_stress_rf30",
            "resnet9_deploy_c_living_stress_rf30",
            None,
            "resnet9_deploy_r1_stress_rf30",
        ),
    }
    seeds3 = (42, 101, 102)
    clean = acc(resolve(by, "resnet9_deploy_r1_stress_rf30"), "baseline_test_acc")
    print(
        f"  clean (baseline_test_acc) per seed: {clean}, mean {mean(clean[s] for s in seeds3):.2f}"
    )
    r1_30 = acc(resolve(by, sweep[30][3]))
    print(f"  R1 @30% spares per seed: {r1_30}, mean {mean(r1_30.values()):.2f}")
    for pct, (d, c, b, r1) in sweep.items():
        line = [f"{pct:>2}%"]
        for lab, spec in (("D", d), ("C'M", c), ("BM", b), ("R1", r1)):
            if spec is None:
                continue
            rows = {s: v for s, v in resolve(by, spec).items() if s in seeds3}
            a = acc(rows)
            stp = mean(r["total_optimizer_steps"] for r in rows.values())
            gap = mean(clean[s] - a[s] for s in a)
            line.append(f"{lab} {mean(a.values()):.2f} (clean-gap {gap:.2f}, steps {stp:.0f})")
        print("  " + " | ".join(line))
    d15_rows, r1_rows = resolve(by, sweep[15][0]), resolve(by, sweep[30][3])
    for sd in seeds3:
        same_maps = [e.get("fault_map_hash") for e in events_of(d15_rows[sd])] == [
            e.get("fault_map_hash") for e in events_of(r1_rows[sd])
        ]
        same_ckpt = d15_rows[sd]["checkpoint_config_hash"] == r1_rows[sd]["checkpoint_config_hash"]
        print(
            f"  pairing check seed {sd}: D@15% vs R1@30% identical per-wave fault-map hashes "
            f"{same_maps}, identical pre-deploy checkpoint {same_ckpt}, dirs "
            f"{d15_rows[sd]['source_dir']} / {r1_rows[sd]['source_dir']}"
        )
    d15 = acc(d15_rows)
    r = paired(d15, r1_30)
    print("  D@15% minus R1@30% (stress_rf15 D, 3 seeds): ", end="")
    show("D-R1@30%", r)
    d_main = acc(resolve(by, "resnet9_deploy_d_eta3e-1+resnet9_final_d"))
    bm = acc(resolve(by, "resnet9_deploy_b_matched_eta5e-1+resnet9_final_b_matched"))
    cmx = acc(resolve(by, "resnet9_deploy_c_living_matched_eta5e-1+resnet9_final_c_living_matched"))
    cl5 = acc(resolve(by, "resnet9_deploy_r1+resnet9_final_r1"), "baseline_test_acc")
    print("  Main test, 5 seeds, 15% spares, gap to each seed's clean accuracy (clean minus arm):")
    for lab, a in (("D", d_main), ("C'-matched", cmx), ("B-matched", bm)):
        gaps = [cl5[s] - a[s] for s in sorted(a)]
        print(
            f"    {lab}: mean {mean(a.values()):.2f}, gap to clean {mean(gaps):.2f} "
            f"(per seed {' '.join(f'{g:.2f}' for g in gaps)})"
        )
    # cumulative stuck fraction and spare capacity
    ev = events_of(next(iter(resolve(by, "resnet9_deploy_d_eta3e-1").values())))
    dens = [float(e["density"]) for e in ev]
    cum, keep = [], 1.0
    for x in dens:
        keep *= 1.0 - x
        cum.append(1.0 - keep)
    print(
        f"  per-wave fault density (D main, seed 42): {dens}; cumulative stuck fraction "
        f"1-prod(1-d): {[round(c, 4) for c in cum]}"
    )
    r1_old = {k: v for k, v in by.items() if k.startswith("resnet9_fault_mixed_remapping_rf")}
    for k in sorted(r1_old):
        s42 = r1_old[k].get(42)
        if s42:
            print(
                f"  (older, different fault config, classic remap) {k} seed 42 end-of-run "
                f"{s42['eor_any']}"
            )
    cl = by.get("resnet9_cont_next_classic_remapping_rf30", {}).get(42)
    c15 = by.get("resnet9_cont_next_classic_remapping", {}).get(42)
    d_cont = resolve(by, "resnet9_cont_d_fixgate:42+resnet9_cont_next_d")
    if cl and c15:
        print(
            f"  continuous regime, seed 42: classic remap 15% {c15['eor_any']}, 30% {cl['eor_any']}, "
            f"D {d_cont[42]['eor_any']} (single seed)"
        )


# ------------------------------------------------------------------ section 4: lifetime
SCEN = {
    "main 3x10%, 15% spares (5 seeds)": {
        "D": "resnet9_deploy_d_eta3e-1+resnet9_final_d",
        "C'M": "resnet9_deploy_c_living_matched_eta5e-1+resnet9_final_c_living_matched",
        "BM": "resnet9_deploy_b_matched_eta5e-1+resnet9_final_b_matched",
        "R1": "resnet9_deploy_r1+resnet9_final_r1",
    },
    "stress 2% spares": {
        "D": "resnet9_deploy_d_stress_rf02",
        "C'M": "resnet9_deploy_c_living_stressM_rf02",
        "BM": "resnet9_deploy_b_stressM_rf02",
        "R1": "resnet9_deploy_r1_stress_rf02",
    },
    "stress 5% spares": {
        "D": "resnet9_deploy_d_stress_rf05",
        "C'M": "resnet9_deploy_c_living_stressM_rf05",
        "BM": "resnet9_deploy_b_stressM_rf05",
        "R1": "resnet9_deploy_r1_stress_rf05",
    },
    "stress 10% spares": {
        "D": "resnet9_deploy_d_stress_rf10",
        "C'M": "resnet9_deploy_c_living_stressM_rf10",
        "BM": "resnet9_deploy_b_stressM_rf10",
        "R1": "resnet9_deploy_r1_stress_rf10",
    },
    "stress 15%/wave": {
        "D": "resnet9_deploy_d_stress_d15",
        "C'M": "resnet9_deploy_c_living_stressM_d15",
        "BM": "resnet9_deploy_b_stressM_d15",
        "R1": "resnet9_deploy_r1_stress_d15",
    },
    "stress 20%/wave": {
        "D": "resnet9_deploy_d_stress_d20",
        "C'M": "resnet9_deploy_c_living_stressM_d20",
        "BM": "resnet9_deploy_b_stressM_d20",
        "R1": "resnet9_deploy_r1_stress_d20",
    },
    "stress 25%/wave": {
        "D": "resnet9_deploy_d_stress_d25",
        "C'M": "resnet9_deploy_c_living_stressM_d25",
        "BM": "resnet9_deploy_b_stressM_d25",
        "R1": "resnet9_deploy_r1_stress_d25",
    },
    "6 small waves (gate-fixed D)": {
        "D": "resnet9_deploy_d_gfix_w6",
        "C'M": "resnet9_deploy_c_living_matchDgfix_w6",
    },
    "8 small waves (gate-fixed D)": {
        "D": "resnet9_deploy_d_gfix_w8",
        "C'M": "resnet9_deploy_c_living_matchDgfix_w8",
    },
    "6x5% waves g5 (gate-fixed D)": {
        "D": "resnet9_deploy_d_g5_gfix",
        "C'M": "resnet9_deploy_c_living_g5_matched",
    },
    "S1 Weibull 5 waves": {
        "D": "resnet9_real_s1_d",
        "C'M": "resnet9_real_s1_c_living_matched",
        "BM": "resnet9_real_s1_b_matched",
        "R1": "resnet9_real_s1_r1",
    },
    "S2 stuck+drift": {
        "D": "resnet9_real_s2_d",
        "C'M": "resnet9_real_s2_c_living_matched",
        "BM": "resnet9_real_s2_b_matched",
        "R1": "resnet9_real_s2_r1",
    },
    "S3 line faults 5 waves": {
        "D": "resnet9_real_s3_d",
        "C'M": "resnet9_real_s3_c_living_matched",
        "BM": "resnet9_real_s3_b_matched",
        "R1": "resnet9_real_s3_r1",
    },
    "S4a 2.0% stuck, 15% spares": {
        "D": "resnet9_calib_s4a_d",
        "BM": "resnet9_calib_s4a_b_matched",
        "R1": "resnet9_calib_s4a_r1",
    },
    "S4b 10.8% stuck, 15% spares": {
        "D": "resnet9_calib_s4b_d",
        "BM": "resnet9_calib_s4b_b_matched",
        "R1": "resnet9_calib_s4b_r1",
    },
    "S4a5 2.0% stuck, 5% spares": {
        "D": "resnet9_calib_s4a5_d",
        "BM": "resnet9_calib_s4a5_b_matched",
        "R1": "resnet9_calib_s4a5_r1",
    },
    "S4b5 10.8% stuck, 5% spares": {
        "D": "resnet9_calib_s4b5_d",
        "BM": "resnet9_calib_s4b5_b_matched",
        "C'M": "resnet9_calib_s4b5_c_living_matched",
        "R1": "resnet9_calib_s4b5_r1",
    },
}


def wave_metrics(row: dict) -> dict:
    ev = events_of(row)
    after = [float(e["test_acc_after"]) for e in ev]
    before = [float(e["test_acc_before"]) for e in ev]
    clean = float(row["baseline_test_acc"])
    floors = {**FLOORS, "Fc5": clean - 5.0, "Fc2": clean - 2.0}
    out = {
        "K": len(after),
        "auc": mean(after),
        "min_after": min(after),
        "worst_dip": min(before),
        "curve": after,
        "last": after[-1],
    }
    for k, f in floors.items():
        n = 0
        for a in after:
            if a < f:
                break
            n += 1
        out[k] = n
    return out


def sec_lifetime(by: dict) -> None:
    print(
        "\n=== 4. Lifetime metric: waves survived above a floor and area under the per-wave curve ==="
    )
    print(
        "  floors F80, F85, Fc5 (= clean-5 pt), Fc2 (= clean-2 pt); clean = baseline_test_acc per run;"
    )
    print(
        "  AUC = mean post-burst test accuracy over the K waves; 'surv' = leading waves >= floor."
    )
    store: dict[str, dict[str, dict[int, dict]]] = {}
    comps: list[tuple[str, str, str, dict]] = []  # scenario, arm, metric, paired result
    for scen, arms in SCEN.items():
        store[scen] = {}
        for arm, spec in arms.items():
            rows = resolve(by, spec)
            store[scen][arm] = {s: wave_metrics(r) for s, r in rows.items()}
        k = {m["K"] for a in store[scen].values() for m in a.values()}
        print(f"\n  [{scen}] K={sorted(k)}")
        for arm, ms in store[scen].items():
            if not ms:
                print(f"    {arm}: no runs")
                continue
            cells = " ".join(
                f"{m}={mean(x[m] for x in ms.values()):.2f}"
                for m in ("F80", "F85", "Fc5", "Fc2", "auc", "min_after", "worst_dip")
            )
            print(f"    {arm} (n={len(ms)}): {cells}")
        for arm, ms in store[scen].items():
            for sd in sorted(ms):
                x = ms[sd]
                print(
                    f"    seed {sd} {arm}: survived F80 {x['F80']} F85 {x['F85']} Fc5 {x['Fc5']} "
                    f"Fc2 {x['Fc2']} of K={x['K']}, AUC {x['auc']:.2f}, last {x['last']:.2f}"
                )
        for arm, ms in store[scen].items():
            if ms:
                k = next(iter(ms.values()))["K"]
                curve = [mean(x["curve"][i] for x in ms.values()) for i in range(k)]
                print(f"    {arm} mean post-burst curve: {' '.join(f'{c:.2f}' for c in curve)}")
        d = store[scen].get("D", {})
        for arm in ("C'M", "BM", "R1"):
            o = store[scen].get(arm)
            if not o or not d:
                continue
            for m in ("auc", "last", "Fc2", "min_after"):
                comps.append(
                    (
                        scen,
                        arm,
                        m,
                        paired({s: v[m] for s, v in d.items()}, {s: v[m] for s, v in o.items()}),
                    )
                )
    prim = [i for i, c in enumerate(comps) if c[2] == "auc" and c[1] in ("C'M", "BM")]
    adj = dict(zip(prim, holm([comps[i][3]["p_t"] for i in prim]), strict=True))
    print(
        f"\n  PRIMARY family: D minus C'M / BM on AUC, {len(prim)} comparisons, Holm on paired-t p."
        "\n  Secondary rows (Fc2, min_after, R1) carry no Holm adjustment and are exploratory."
    )
    for i, (scen, arm, m, r) in enumerate(comps):
        if arm == "R1" and m != "auc":
            continue
        tag = "PRIMARY" if i in prim else "secondary"
        show(f"[{tag}] {scen} | D-{arm} | {m}", r, adj.get(i))


# ------------------------------------------------------------------ section 5: closed loop
def final_of(path: Path) -> float | None:
    f = path / "events.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return float(d["summary"]["final_test_acc"]) if d.get("complete") else None


def loc_of(path: Path) -> dict:
    d = json.loads((path / "events.json").read_text())
    locs = [e["localize"] for e in d["events"] if "localize" in e]
    rec = [x["found_new_visible"] / x["truth_new_visible"] for x in locs if x["truth_new_visible"]]
    reads, sreads = sum(x["reads"] for x in locs), sum(x["shadow_reads"] for x in locs)
    return {
        "recall_min": min(rec) if rec else 1.0,
        "reads_ratio": reads / sreads if sreads else float("nan"),
        "fallback": sum(x["fallback_tiles"] for x in locs) / sum(x["tiles"] for x in locs),
    }


def sec_closed_loop() -> None:
    print("\n=== 5. Closed loop X: D on a probe-estimated fault map vs D on the oracle map ===")
    box = RES / "spectral" / "x_loop_box" / "results"
    ext = RES / "spectral" / "x_loop_ext" / "results"
    live9 = RES / "_live9"
    seeds = (42, 101, 102, 103, 104)

    def where(name: str) -> Path:
        return box / name if (box / name / "events.json").exists() else ext / name

    for regime in ("dense", "sparse"):
        for sig in ("0.02", "0.04"):
            gaps, recs, rr, fb, floors = [], [], [], [], []
            for s in seeds:
                est = where(f"x_{regime}_d_xadaptive_s{sig}_seed{s}")
                if regime == "dense":
                    ora = live9 / f"resnet9_real_s1_d_seed{s}"
                    if not (ora / "events.json").exists():
                        ora = where(f"x_dense_d_xoracle_seed{s}")
                else:
                    ora = where(f"x_sparse_d_xoracle_seed{s}")
                fe, fo = final_of(est), final_of(ora)
                if fe is None or fo is None:
                    continue
                gaps.append(fe - fo)
                lc = loc_of(est)
                recs.append(lc["recall_min"])
                rr.append(lc["reads_ratio"])
                fb.append(lc["fallback"])
                if regime == "sparse":
                    fl = final_of(where(f"x_sparse_r0_xoracle_seed{s}"))
                    floors.append((fo, fl))
            if regime == "sparse":
                print(
                    f"    sparse sigma={sig} reads ratio per seed {list(seeds)}: "
                    f"{' '.join(f'{x:.4f}' for x in rr)}; min wave recall per seed "
                    f"{' '.join(f'{x:.3f}' for x in recs)}"
                )
            r = paired({i: g for i, g in enumerate(gaps)}, {i: 0.0 for i in range(len(gaps))})
            print(
                f"  ResNet-9 {regime} sigma={sig}: n={len(gaps)} gap(est-oracle) mean "
                f"{fmt(mean(gaps))} range [{fmt(min(gaps))},{fmt(max(gaps))}] p_t={r['p_t']:.3f}; "
                f"min wave recall {min(recs):.3f}; reads ratio {mean(rr):.3f} (max {max(rr):.3f}); "
                f"fallback frac {mean(fb):.3f}"
            )
    est_d = {s: final_of(where(f"x_dense_d_xadaptive_s0.02_seed{s}")) for s in seeds}
    est_c = {s: final_of(where(f"x_dense_c_living_matched_xadaptive_s0.02_seed{s}")) for s in seeds}
    diffs = [est_d[s] - est_c[s] for s in seeds if est_d[s] is not None and est_c[s] is not None]
    print(
        f"  dense 2%: D-est minus C'M-est (both estimated, matched): {' '.join(fmt(x) for x in diffs)} "
        f"mean {fmt(mean(diffs))}"
    )
    rerun = final_of(box / "x_dense_d_xoracle_seed42")
    first = final_of(live9 / "resnet9_real_s1_d_seed42")
    print(
        f"  oracle reproducibility, dense D seed 42: rerun {rerun} vs _live9 {first} "
        f"(difference {fmt(rerun - first)}); registered gap bars: dense -1.0, sparse -0.5 pt; "
        f"registered reads bar: <= 0.25 of full readout on every seed (0.35 for sigma 4% and ViT)"
    )
    for regime in ("dense", "sparse"):
        gaps = []
        for s in seeds:
            e = final_of(where(f"x_vit_{regime}_d_xadaptive_s0.02_seed{s}"))
            o = final_of(where(f"x_vit_{regime}_d_xoracle_seed{s}"))
            if e is not None and o is not None:
                gaps.append(e - o)
        print(
            f"  ViT-tiny {regime} sigma=0.02: n={len(gaps)} gap mean {fmt(mean(gaps))} "
            f"range [{fmt(min(gaps))},{fmt(max(gaps))}]"
        )


def main() -> None:
    by = build_db()
    sec_overwrite_and_nets(by)
    sec_tuning(by)
    sec_canary(by)
    sec_heldout(by)
    sec_spares(by)
    sec_lifetime(by)
    sec_closed_loop()


if __name__ == "__main__":
    main()

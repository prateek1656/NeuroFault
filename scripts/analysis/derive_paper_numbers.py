"""Re-derive every paper number from raw result files (results/**/events.json).

Reads only ``results/`` (never ``results/spectral``).  Writes three files into
``results/derived/``:

* ``canonical_runs.csv``  one row per (run_name, seed) with all raw metrics.
* ``claims.csv``          one row per reported number, with the derived value.
* ``canary_selection.csv`` learning-rate sweep with test and canary scores.

Deterministic: no randomness, rows sorted.  Run:

    uv run python scripts/analysis/derive_paper_numbers.py

Conventions
-----------
* Deployment runs: ``final_test_acc`` is the test accuracy after the LAST wave's burst.
* Continuous runs: ``final_test_acc`` is the accuracy right after wave 3 (batch 15600).
  ``end_of_run_test_acc`` is the accuracy after the last training batch (newer runs only).
  ``eor_any`` falls back to the last logged point of metrics.json (batch 19100 for
  continuous event runs, 19400 for the classic metrics-only runs).
* Duplicate run names (same name and seed in several dirs): the canonical copy is the one
  pinned in ``results/derived/canonical_pins.csv`` (run_name, seed, source_dir) when that
  file lists it; otherwise the first complete copy in lexicographic path order.  The pin
  file is an input and is never overwritten by this script, so the choice does not depend on
  file timestamps.  The row carries a duplicate flag.  A claim can force a copy with
  ``name@_live5``.
* Arm spec in claims: ``part+part+...``; a part is ``name[@live][:seed,seed]``.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "results" / "derived"
SEED_RE = re.compile(r"^(?P<run>.+)_seed(?P<seed>\d+)$")
MAX_WAVES = 10
PINS = OUT / "canonical_pins.csv"


# --------------------------------------------------------------------------
# Part 1: collect runs
# --------------------------------------------------------------------------
def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _last_metric(run_dir: Path) -> tuple[float | None, int | None, float | None]:
    """Last logged test accuracy, its batch, and the mean of the last 5 logs (metrics.json)."""
    m = _load(run_dir / "metrics.json")
    if not isinstance(m, dict) or not m.get("accuracy") or not m.get("batch"):
        return None, None, None
    acc = [float(x) for x in m["accuracy"]]
    return acc[-1], int(m["batch"][-1]), sum(acc[-5:]) / len(acc[-5:])


def _blank_waves() -> dict:
    out = {}
    for i in range(1, MAX_WAVES + 1):
        out[f"w{i}_acc_after"] = None
        out[f"w{i}_steps"] = None
    return out


def _base_row(run_dir: Path) -> dict:
    mt = SEED_RE.match(run_dir.name)
    last, batch, last5 = _last_metric(run_dir)
    return {
        "run_name": mt["run"],
        "seed": int(mt["seed"]),
        "source_dir": str(run_dir.relative_to(ROOT)),
        "regime": None,
        "policy": None,
        "complete": False,
        "final_test_acc": None,
        "end_of_run_test_acc": None,
        "metrics_last_acc": last,
        "metrics_last_batch": batch,
        "metrics_last5_mean": last5,
        "lifetime_mean": None,
        "worst_dip": None,
        "total_burst_batches": None,
        "total_optimizer_steps": None,
        "total_remapped": None,
        "baseline_test_acc": None,
        "checkpoint_final_test_acc": None,
        "checkpoint_config_hash": None,
        "n_waves": 0,
        "canary_last": None,
        "canary_mean": None,
        "max_unremapped": None,
        "plateau_min_gain_mean": None,
        **_blank_waves(),
    }


def _event_row(run_dir: Path, ev: dict) -> dict:
    row = _base_row(run_dir)
    s = ev.get("summary") or {}
    events = ev.get("events") or []
    row.update(
        regime=ev.get("regime"),
        policy=ev.get("policy"),
        complete=bool(ev.get("complete")),
        final_test_acc=s.get("final_test_acc"),
        end_of_run_test_acc=s.get("end_of_run_test_acc"),
        lifetime_mean=s.get("lifetime_mean_test_acc"),
        worst_dip=s.get("worst_dip_test_acc"),
        total_burst_batches=s.get("total_burst_batches"),
        total_remapped=s.get("total_remapped"),
        baseline_test_acc=ev.get("baseline_test_acc"),
        checkpoint_final_test_acc=ev.get("checkpoint_final_test_acc"),
        checkpoint_config_hash=ev.get("checkpoint_config_hash"),
        n_waves=len(events),
    )
    row["total_optimizer_steps"] = sum(e.get("optimizer_steps") or 0 for e in events)
    canary = [e["canary_acc_after"] for e in events if e.get("canary_acc_after") is not None]
    if canary:
        row["canary_last"] = canary[-1]
        row["canary_mean"] = sum(canary) / len(canary)
    unrem = [
        e["unremapped_effective_stuck"]
        for e in events
        if e.get("unremapped_effective_stuck") is not None
    ]
    row["max_unremapped"] = max(unrem) if unrem else None
    gains = [e["plateau_min_gain"] for e in events if e.get("plateau_min_gain") is not None]
    row["plateau_min_gain_mean"] = sum(gains) / len(gains) if gains else None
    for i, e in enumerate(events[:MAX_WAVES], start=1):
        row[f"w{i}_acc_after"] = e.get("test_acc_after")
        row[f"w{i}_steps"] = e.get("optimizer_steps", e.get("burst_batches"))
    return row


def _metrics_row(run_dir: Path) -> dict | None:
    """Runs that only have metrics.json (classic Block-2 arms, no wave events)."""
    row = _base_row(run_dir)
    if row["metrics_last_acc"] is None:
        return None
    row.update(regime="metrics_only", policy="classic", complete=True)
    row["end_of_run_test_acc"] = row["metrics_last_acc"]
    return row


def _candidate_dirs() -> list[Path]:
    """Seeded run dirs under results/ (depth 1 or 2); never descends into spectral."""
    out: list[Path] = []
    for top in sorted(RESULTS.iterdir()):
        if not top.is_dir() or "spectral" in top.name:
            continue
        cands = [top] + [d for d in sorted(top.iterdir()) if d.is_dir()]
        out += [d for d in cands if SEED_RE.match(d.name)]
    return out


def _fingerprint(r: dict) -> tuple:
    last = r["metrics_last_acc"] if r["final_test_acc"] is None else None
    return (
        r["final_test_acc"],
        r["total_burst_batches"],
        r["checkpoint_config_hash"],
        r["n_waves"],
        last,
    )


def _load_pins() -> dict[tuple[str, int], str]:
    """(run_name, seed) -> pinned canonical source_dir, from the committed pin file."""
    if not PINS.exists():
        return {}
    with PINS.open(newline="") as fh:
        return {(r["run_name"], int(r["seed"])): r["source_dir"] for r in csv.DictReader(fh)}


def collect() -> tuple[list[dict], list[dict]]:
    """Return (canonical rows, all rows).  Duplicates flagged.

    Canonical copy: the pinned one if listed in canonical_pins.csv and present, else the first
    complete copy in lexicographic path order (deterministic, independent of file times).
    """
    allrows: list[dict] = []
    for d in _candidate_dirs():
        if (d / "events.json").exists():
            ev = _load(d / "events.json")
            if isinstance(ev, dict):
                allrows.append(_event_row(d, ev))
        elif d.name.startswith("resnet9_") and (d / "metrics.json").exists():
            row = _metrics_row(d)
            if row is not None:
                allrows.append(row)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in allrows:
        groups[(r["run_name"], r["seed"])].append(r)
    pins = _load_pins()
    canon = []
    for key, rs in sorted(groups.items()):
        pinned = [r for r in rs if r["source_dir"] == pins.get(key)]
        ordered = sorted(rs, key=lambda r: (not r["complete"], r["source_dir"]))
        pick = dict((pinned or ordered)[0])
        pick["n_copies"] = len(rs)
        pick["all_sources"] = ";".join(sorted(r["source_dir"] for r in rs))
        if len(rs) == 1:
            pick["duplicate_flag"] = ""
        else:
            same = len({_fingerprint(r) for r in rs}) == 1
            pick["duplicate_flag"] = "identical" if same else "DIFFERENT"
        eor = pick["end_of_run_test_acc"]
        uses_log = pick["regime"] in ("continuous", "metrics_only")
        fallback = pick["metrics_last_acc"] if uses_log else None
        pick["eor_any"] = eor if eor is not None else fallback
        pick["eor_source"] = (
            "" if pick["eor_any"] is None else "summary" if eor is not None else "metrics_last"
        )
        canon.append(pick)
    return canon, allrows


# --------------------------------------------------------------------------
# Part 2: claims
# --------------------------------------------------------------------------
def _live_of(row: dict) -> str:
    return row["source_dir"].split("/")[1]


class DB:
    """Lookup of complete runs by name and seed, including per-dir copies."""

    def __init__(self, canon: list[dict], allrows: list[dict]):
        self.by: dict[str, dict[int, dict]] = defaultdict(dict)
        self.by_live: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
        for r in canon:
            if r["complete"]:
                self.by[r["run_name"]][r["seed"]] = r
        for r in allrows:
            if r["complete"]:
                self.by_live[(r["run_name"], _live_of(r))][r["seed"]] = r

    def resolve(self, spec: str) -> dict[int, dict]:
        """Map seed -> row for an arm spec (parts joined by '+')."""
        out: dict[int, dict] = {}
        for part in spec.split("+"):
            name, _, seeds = part.partition(":")
            name, _, live = name.partition("@")
            rows = self.by_live.get((name, live), {}) if live else self.by.get(name, {})
            want = {int(s) for s in seeds.split(",")} if seeds else None
            for seed, row in rows.items():
                if (want is None or seed in want) and seed not in out:
                    out[seed] = row
        return out


def _fmt(x: float | None, nd: int = 2) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _reduce(stat: str, xs: list[float], seeds: list[int]) -> float | None:
    if not xs:
        return None
    if stat == "mean":
        return st.mean(xs)
    if stat == "sd":
        return st.stdev(xs) if len(xs) > 1 else None
    if stat == "min":
        return min(xs)
    if stat == "max":
        return max(xs)
    if stat == "wins":
        return float(sum(x > 1e-9 for x in xs))
    if stat == "n":
        return float(len(xs))
    if stat.startswith("seed:"):
        s = int(stat.split(":")[1])
        return xs[seeds.index(s)] if s in seeds else None
    raise ValueError(stat)


class Claims:
    def __init__(self, db: DB):
        self.db = db
        self.rows: list[dict] = []

    def _add(self, row: dict) -> None:
        self.rows.append(row)

    def q(
        self,
        cid,
        desc,
        a,
        key="final_test_acc",
        stat="mean",
        b=None,
        seeds=None,
        key_b=None,
        note="",
    ):
        """One number.  Single arm if b is None, else paired (A minus B) per seed.

        stat: mean, sd, min, max, wins, n, seed:N, ratio_pct (mean(a)/mean(b)*100).
        """
        ra = self.db.resolve(a)
        rb = self.db.resolve(b) if b else None
        common = sorted(set(ra) & set(rb)) if rb is not None else sorted(ra)
        if seeds:
            common = [s for s in common if s in seeds]
        va = {s: ra[s].get(key) for s in common if ra[s].get(key) is not None}
        vb = {}
        if rb is not None:
            vb = {s: rb[s].get(key_b or key) for s in common if rb[s].get(key_b or key) is not None}
            common = [s for s in common if s in va and s in vb]
        else:
            common = [s for s in common if s in va]
        if rb is not None:
            vals = [va[s] - vb[s] for s in common]
        else:
            vals = [va[s] for s in common]
        if stat == "ratio_pct" and rb is not None and common:
            derived = 100 * st.mean(va[s] for s in common) / st.mean(vb[s] for s in common)
        else:
            derived = _reduce(stat, vals, common)
        row = {
            "claim_id": cid,
            "description": desc,
            "arms": f"{a} MINUS {b}" if b else a,
            "measure": key,
            "stat": stat,
            "seeds": " ".join(map(str, common)),
            "per_seed": " ".join(_fmt(v) for v in vals),
            "mean": _fmt(st.mean(vals)) if vals else "",
            "sd": _fmt(st.stdev(vals)) if len(vals) > 1 else "",
            "paired_diffs": " ".join(f"{v:+.2f}" for v in vals) if rb is not None else "",
            "n_wins": (f"{sum(v > 1e-9 for v in vals)}/{len(vals)}" if rb is not None else ""),
            "derived_value": _fmt(derived) if derived is not None else "NO-SOURCE",
            "source_runs": f"{a} || {b}" if b else a,
            "source_dirs": ";".join(sorted({ra[s]["source_dir"] for s in common}))
            if common
            else "",
            "same_checkpoint": self._same_ckpt(ra, rb, common),
            "note": note,
        }
        self._add(row)

    @staticmethod
    def _same_ckpt(ra, rb, common) -> str:
        if rb is None or not common:
            return ""
        ok = all(ra[s]["checkpoint_config_hash"] == rb[s]["checkpoint_config_hash"] for s in common)
        return "yes" if ok else "NO"

    def custom(self, cid, desc, derived, source, note=""):
        """A number not taken straight from one arm (arithmetic or config facts)."""
        row = {
            "claim_id": cid,
            "description": desc,
            "arms": "",
            "measure": "custom",
            "stat": "",
            "seeds": "",
            "per_seed": "",
            "mean": "",
            "sd": "",
            "paired_diffs": "",
            "n_wins": "",
            "same_checkpoint": "",
            "derived_value": "NO-SOURCE" if derived is None else _fmt(derived, 3),
            "source_runs": source,
            "source_dirs": "",
            "note": note,
        }
        self._add(row)


# --- run-name shorthands ---------------------------------------------------
R9 = "resnet9_deploy_"
D3 = R9 + "d_eta3e-1"  # D, tuned rate 0.3
B5, C5 = R9 + "b_eta5e-1", R9 + "c_living_eta5e-1"
BM, CM = R9 + "b_matched_eta5e-1", R9 + "c_living_matched_eta5e-1"
D5 = f"{D3}+resnet9_final_d"  # five seeds: 42,101,102 + 103,104
BM5 = f"{BM}+resnet9_final_b_matched"
CM5 = f"{CM}+resnet9_final_c_living_matched"
R1_5 = f"{R9}r1+resnet9_final_r1"
CLEAN_KEY = "baseline_test_acc"


def _net(prefix: str, final: str) -> dict:
    """Spec strings for VGG6 / ViT (net2 / net3 tuned sweep + final seeds 103,104)."""
    n = prefix
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


VGG = _net("vgg6_net2", "vgg6_final_deploy")
VIT = _net("vit_net3", "vit_final_deploy")
VGG["r1"] = "vgg6_net2_deploy_r1+vgg6_final_deploy_r1"
VIT["r1"] = "vit_net3_deploy_r1+vit_final_deploy_r1"
VGG["bm"] = "vgg6_net2_deploy_b_matched_best+vgg6_final_deploy_b_matched"
VIT["bm"] = "vit_net3_deploy_b_matched_best+vit_final_deploy_b_matched"
VGG["cm"] = "vgg6_net2_deploy_c_living_matched_best+vgg6_final_deploy_c_living_matched"
VIT["cm"] = "vit_net3_deploy_c_living_matched_best+vit_final_deploy_c_living_matched"
VGG["d"] = "vgg6_net2_deploy_d_eta0.3:42+vgg6_net2_deploy_d_best+vgg6_final_deploy_d"
VIT["d"] = "vit_net3_deploy_d_eta0.3:42+vit_net3_deploy_d_best+vit_final_deploy_d"


def build_claims(db: DB) -> list[dict]:
    c = Claims(db)
    clean_and_basics(c)
    findings_1_to_5(c)
    finding_6(c)
    finding_7(c)
    finding_8_9(c)
    finding_10(c)
    finding_11(c)
    findings_12_13(c)
    finding_14(c)
    finding_15(c)
    sa_classic_claims(c)
    section_5(c)
    config_facts(c)
    prereg_claims(c)
    canary_claims(c)
    return c.rows


def clean_and_basics(c: Claims) -> None:
    for s in (42, 101, 102):
        c.q(
            f"intro.clean.s{s}",
            f"clean chip seed {s} (deployment base net)",
            R9 + "r0",
            CLEAN_KEY,
            f"seed:{s}",
        )
    c.q("intro.clean.mean3", "clean chip, 3-seed mean", R9 + "r0", CLEAN_KEY)
    c.q(
        "intro.clean.mean5",
        "clean chip, 5-seed mean (F11 table uses 89.0)",
        f"{R9}r0+resnet9_final_r1",
        CLEAN_KEY,
    )


def findings_1_to_5(c: Claims) -> None:
    c.q("F1.r0", "R0 do nothing, 3 seeds", R9 + "r0")
    c.q("F1.r1", "R1 remap only, 3 seeds", R9 + "r1")
    c.q("F2.noburst", "D without bursts, 3 seeds", R9 + "d_noburst")
    c.q("F2.d_old", "D with bursts, old rate 0.005", R9 + "d")
    c.q("F3.gap_old", "D minus C' at old rate", R9 + "d", b=R9 + "c_living", stat="mean")
    c.q("F3.d_old", "D old rate", R9 + "d")
    c.q("F3.cp_old", "C' old rate (living, 780 steps)", R9 + "c_living")
    c.q("F3.b_old", "B old rate", R9 + "b")
    c.q("F3.cp_old2", "C' old rate", R9 + "c_living")
    c.q("F3.d_old2", "D old rate", R9 + "d")
    c.q("F3.b_tuned", "B tuned rate 0.5", B5)
    c.q("F3.cp_tuned", "C' tuned rate 0.5", C5)
    c.q("F3.d_tuned", "D tuned rate 0.3", D3)
    c.q("F3.b_sweep_lo", "B at rate 0.001, seed 42 only", R9 + "b_eta1e-3")
    c.q("F3.b_sweep_hi", "B at rate 0.5, seed 42 only", B5, stat="seed:42")
    c.q("F3.b_swing3", "B tuned minus B old, 3 seeds", B5, b=R9 + "b")
    c.q("F3.b_swing1", "B rate 0.5 minus 0.001, seed 42", B5, b=R9 + "b_eta1e-3", seeds=[42])
    # ---- Finding 4
    for lab, arm_b in (("bm", BM5), ("cm", CM5)):
        c.q(f"F4.{lab}.mean5", f"{lab} mean, 5 seeds", arm_b)
        c.q(f"F4.d_minus_{lab}.mean5", f"D minus {lab}, 5 seeds", D5, b=arm_b)
        c.q(f"F4.d_minus_{lab}.wins5", f"D wins vs {lab} (of 5)", D5, b=arm_b, stat="wins")
    for s in (42, 101, 102, 103, 104):
        c.q(f"F4.d_minus_bm.s{s}", f"D minus B-matched seed {s}", D5, b=BM5, stat=f"seed:{s}")
    for s in (42, 101, 102, 103, 104):
        c.q(f"F4.d_minus_cm.s{s}", f"D minus C'-matched seed {s}", D5, b=CM5, stat=f"seed:{s}")
    c.q("F4.d5", "D mean, 5 seeds", D5)
    c.q("F4.cm3", "C'-matched mean, 3 seeds", CM)
    c.q("F4.d_minus_cm.mean3", "D minus C'-matched, 3 seeds", D3, b=CM)
    c.q("F4.d_minus_bm.mean3", "D minus B-matched, 3 seeds", D3, b=BM)
    c.q("F4.steps.d", "D burst steps, mean of seeds 42-102", D3, "total_burst_batches")
    c.q("F4.steps.cm", "C'-matched burst steps, mean of seeds 42-102", CM, "total_burst_batches")
    c.q("F4.steps.d.103_104", "D steps seeds 103,104", "resnet9_final_d", "total_burst_batches")
    c.q(
        "F4.steps.cm.103_104",
        "C'-matched steps seeds 103,104",
        "resnet9_final_c_living_matched",
        "total_burst_batches",
    )
    c.custom("F4.signtest5", "one-sided sign test, 5 of 5 (0.5**5)", 0.5**5, "arithmetic")
    c.custom("F4.signtest3", "one-sided sign test, 3 of 3 (0.5**3)", 0.5**3, "arithmetic")
    # ---- Finding 5
    nopace = "resnet9_final_d_nopace_matched"
    c.q(
        "F5.pace_unmatched.BD",
        "D minus no-pacing (run BD, old matching 1000 steps)",
        D3,
        b=R9 + "d_nopace_etaBD",
    )
    c.q("F5.pace_unmatched.BD2", "D minus no-pacing (run BD2)", D3, b=R9 + "d_nopace_etaBD2")
    c.q(
        "F5.pace_unmatched.wins",
        "D wins vs no-pacing BD2 (of 3)",
        D3,
        b=R9 + "d_nopace_etaBD2",
        stat="wins",
    )
    c.q("F5.pace_unmatched.steps_d", "D steps", D3, "total_burst_batches")
    c.q(
        "F5.pace_unmatched.steps_np",
        "no-pacing steps (BD2)",
        R9 + "d_nopace_etaBD2",
        "total_burst_batches",
    )
    c.q("F5.pace_matched.mean", "D minus step-matched no-pacing", D3, b=nopace)
    for s in (42, 101, 102):
        c.q(
            f"F5.pace_matched.s{s}",
            f"D minus matched no-pacing seed {s}",
            D3,
            b=nopace,
            stat=f"seed:{s}",
        )
    c.q("F5.pace_matched.wins", "D wins vs matched no-pacing (of 3)", D3, b=nopace, stat="wins")
    c.q(
        "F5.pace_matched.steps_d",
        "D steps",
        D3,
        "total_burst_batches",
        stat="seed:42",
        note="D seed 42 steps",
    )
    c.q(
        "F5.pace_matched.steps_np",
        "matched no-pacing steps seed 42",
        nopace,
        "total_burst_batches",
        stat="seed:42",
    )
    c.q(
        "F5.pace_matched.steps_d102", "D steps seed 102", D3, "total_burst_batches", stat="seed:102"
    )
    c.q(
        "F5.pace_matched.steps_np102",
        "matched no-pacing steps seed 102",
        nopace,
        "total_burst_batches",
        stat="seed:102",
    )
    c.q("F5.rank.BD", "D minus random order (BD)", D3, b=R9 + "d_nosal_etaBD")
    c.q("F5.rank.BD2", "D minus random order (BD2)", D3, b=R9 + "d_nosal_etaBD2")
    c.q(
        "F5.rank.steps_BD",
        "random-order steps (BD) mean",
        R9 + "d_nosal_etaBD",
        "total_burst_batches",
    )
    c.q(
        "F5.rank.steps_BD2",
        "random-order steps (BD2) mean",
        R9 + "d_nosal_etaBD2",
        "total_burst_batches",
    )
    c.q("F5.mag.BD", "D minus weight-size order (BD)", D3, b=R9 + "d_mag_etaBD")
    c.q("F5.mag.BD2", "D minus weight-size order (BD2)", D3, b=R9 + "d_mag_etaBD2")
    c.q("F5.mag.wins.BD", "D wins vs weight-size BD (of 3)", D3, b=R9 + "d_mag_etaBD", stat="wins")
    c.q(
        "F5.mag.wins.BD2",
        "D wins vs weight-size BD2 (of 3)",
        D3,
        b=R9 + "d_mag_etaBD2",
        stat="wins",
    )
    c.q("F5.mag.steps_BD", "weight-size steps (BD)", R9 + "d_mag_etaBD", "total_burst_batches")
    c.q("F5.mag.steps_BD2", "weight-size steps (BD2)", R9 + "d_mag_etaBD2", "total_burst_batches")
    c.q("F5.nobn.gap", "D minus no-BN-fix (old rate)", R9 + "d", b=R9 + "d_nobn")
    c.q("F5.nobn.d", "D old rate", R9 + "d")
    c.q("F5.nobn.abl", "no-BN-fix old rate", R9 + "d_nobn")
    c.q("F5.nobn.steps_d", "D old-rate steps", R9 + "d", "total_burst_batches")
    c.q("F5.nobn.steps_abl", "no-BN steps", R9 + "d_nobn", "total_burst_batches")
    c.q("F5.noadapt.abl", "no-adapt old rate", R9 + "d_noadapt")
    c.q("F5.noadapt.d", "D old rate", R9 + "d")
    c.q("F5.head.rank", "heading: ranking helps about 1.7 (BD2)", D3, b=R9 + "d_nosal_etaBD2")
    c.q("F5.head.pace", "heading: pacing about 0.7 matched", D3, b=nopace)
    c.q(
        "F5.noise.nopace",
        "run-to-run noise BD2 minus BD (nopace)",
        R9 + "d_nopace_etaBD2",
        b=R9 + "d_nopace_etaBD",
    )
    c.q(
        "F5.noise.mag",
        "run-to-run noise BD2 minus BD (mag), max |seed diff| 1.22",
        R9 + "d_mag_etaBD2",
        b=R9 + "d_mag_etaBD",
        stat="max",
    )


GATE = ("w6", "w8")


def finding_6(c: Claims) -> None:
    cases = (
        ("rf02", "2% spare"),
        ("rf05", "5% spare"),
        ("rf10", "10% spare"),
        ("d15", "15% faults/wave"),
        ("d20", "20% faults/wave"),
        ("d25", "25% faults/wave"),
        ("w6", "6 small waves (old gate)"),
        ("w8", "8 small waves (old gate)"),
    )
    for k, lab in cases:
        d, cm = R9 + f"d_stress_{k}", R9 + f"c_living_stressM_{k}"
        c.q(f"F6.{k}.gap", f"D minus C'-matched, {lab}", d, b=cm)
        c.q(f"F6.{k}.wins", f"D wins vs C'-matched, {lab} (of 3)", d, b=cm, stat="wins")
        c.q(
            f"F6.{k}.steps",
            f"D steps vs C'-matched steps, {lab}",
            d,
            "total_burst_batches",
            note="compare with next row",
        )
        c.q(f"F6.{k}.steps_cm", f"C'-matched steps, {lab}", cm, "total_burst_batches")
    c.q(
        "F6.rf05.minseed",
        "5% spare: worst per-seed gap",
        R9 + "d_stress_rf05",
        b=R9 + "c_living_stressM_rf05",
        stat="min",
    )
    # gate-fix rows
    for w in ("w6", "w8"):
        dg = R9 + f"d_gfix_{w}"
        cmatch = R9 + f"c_living_matchDgfix_{w}"
        cold = R9 + f"c_living_stressM_{w}"
        c.q(f"F6.gfix.{w}.d", f"gate-fixed D, {w}", dg)
        c.q(f"F6.gfix.{w}.cp", f"C' with D-gfix steps ('matchDgfix'), {w}", cmatch)
        c.q(f"F6.gfix.{w}.gap", f"D-gfix minus C' matched to D-gfix steps, {w}", dg, b=cmatch)
        c.q(
            f"F6.gfix.{w}.wins",
            f"D-gfix wins vs matchDgfix C' (of 3), {w}",
            dg,
            b=cmatch,
            stat="wins",
        )
        c.q(f"F6.gfix.{w}.steps_d", f"D-gfix steps, {w}", dg, "total_burst_batches")
        c.q(f"F6.gfix.{w}.steps_cmatch", f"matchDgfix C' steps, {w}", cmatch, "total_burst_batches")
        c.q(
            f"F6.gfix.{w}.steps_cold",
            f"old C'-matched (stressM) steps, {w}",
            cold,
            "total_burst_batches",
        )
        c.q(
            f"F6.gfix.{w}.gap_vs_old", f"D-gfix minus OLD stressM C' (fewer steps), {w}", dg, b=cold
        )
    c.q("F6.gfix.w8.improve", "D 8 waves old gate -> gate-fix: gate-fix value", R9 + "d_gfix_w8")
    c.q("F6.gfix.w8.old", "D 8 waves old gate", R9 + "d_stress_w8")


def finding_7(c: Claims) -> None:
    dcont = "resnet9_cont_d_fixgate:42+resnet9_cont_next_d"
    ccont = "resnet9_cont_c_living:42+resnet9_cont_next_c_living"
    cmcont = "resnet9_cont_next_c_living_matched"
    k = "eor_any"
    c.q(
        "F7.nomit",
        "classic no mitigation seed 42 (batch 19400)",
        "resnet9_cont_next_classic_no_mitigation",
        k,
    )
    c.q(
        "F7.nomit_old",
        "older classic no-mit run (_live2, secondary drift fault)",
        "resnet9_fault_mixed_no_mitigation",
        k,
    )
    c.q("F7.remap15", "classic remap 15% spares seed 42", "resnet9_cont_next_classic_remapping", k)
    c.q(
        "F7.remap15_old",
        "older classic remap 15% (_live2, secondary drift fault)",
        "resnet9_fault_mixed_remapping",
        k,
    )
    c.q(
        "F7.remap30",
        "classic remap 30% spares seed 42",
        "resnet9_cont_next_classic_remapping_rf30",
        k,
    )
    c.q(
        "F7.remap30_old",
        "older classic remap 30% (_live2)",
        "resnet9_fault_mixed_remapping_rf30",
        k,
    )
    c.q("F7.cp_eor", "C' living (unmatched), end of run, 3 seeds", ccont, key=k)
    c.q("F7.cp_final", "C' living, value right after wave 3, 3 seeds", ccont)
    c.q("F7.cp_final.s42", "C' living seed 42, value right after wave 3", ccont, stat="seed:42")
    c.q("F7.cm_eor", "C'-matched, end of run, 3 seeds", cmcont, key=k)
    for s in (42, 101, 102):
        c.q(f"F7.cm.s{s}", f"C'-matched end of run seed {s}", cmcont, key=k, stat=f"seed:{s}")
    c.q("F7.d_eor", "D (gate fixed), end of run, 3 seeds", dcont, key=k)
    c.q("F7.d_final", "D (gate fixed), value right after wave 3, 3 seeds", dcont)
    for s in (42, 101, 102):
        c.q(f"F7.d.s{s}.eor", f"D end of run seed {s}", dcont, key=k, stat=f"seed:{s}")
    c.q("F7.d.s42.wave3", "D seed 42 value after wave 3 (batch 15600)", dcont, stat="seed:42")
    for s in (101, 102):
        c.q(
            f"F7.d.s{s}.wave3",
            f"D seed {s} value after wave 3 (summary.final_test_acc)",
            dcont,
            stat=f"seed:{s}",
            note="raw final_test_acc is the wave-3 value, not the end-of-run value",
        )
    c.q("F7.gap_cm_eor", "D minus C'-matched, end of run, 3 seeds", dcont, b=cmcont, key=k)
    for s in (42, 101, 102):
        c.q(
            f"F7.gap_cm.s{s}.eor",
            f"D minus C'-matched seed {s}, end of run",
            dcont,
            b=cmcont,
            key=k,
            stat=f"seed:{s}",
        )
    c.q(
        "F7.gap_cm_mixed",
        "D minus C'-matched mixing D wave-3 (seed 42) and eor",
        dcont,
        b=cmcont,
        key="final_test_acc",
        key_b=k,
        stat="seed:42",
        note="seed 42 pair: D final 89.12 minus C'M eor 87.66",
    )
    c.q(
        "F7.gap_cm.wins",
        "D beats C'-matched (end of run) in n of 3",
        dcont,
        b=cmcont,
        key=k,
        stat="wins",
    )
    c.q("F7.gap_cp", "D minus original C' (unmatched), end of run", dcont, b=ccont, key=k)
    c.q(
        "F7.d_vs_remap30",
        "D (eor, 3 seeds) minus classic remap 30%",
        dcont,
        b="resnet9_cont_next_classic_remapping_rf30:42",
        key=k,
        seeds=[42],
    )
    c.q("F7.d_firstbug", "first D run (gate bug) value right after wave 3", "resnet9_cont_d")
    c.q("F7.d_firstbug_eor", "first D run (gate bug) at end of training", "resnet9_cont_d", key=k)
    c.q("F7.d_firstbug_steps", "first D run bursts", "resnet9_cont_d", "total_burst_batches")
    c.q(
        "F7.last_batch.d",
        "last logged batch of continuous D run",
        dcont,
        "metrics_last_batch",
        stat="mean",
    )
    c.q(
        "F7.last_batch.classic",
        "last logged batch of classic run",
        "resnet9_cont_next_classic_no_mitigation",
        "metrics_last_batch",
    )


def finding_8_9(c: Claims) -> None:
    rows = (
        ("clu", "Clustered faults"),
        ("rf05", "Spare cells 5%"),
        ("rf10", "Spare cells 10%"),
        ("rf30", "Spare cells 30%"),
        ("g1", "Light faults"),
    )
    for k, lab in rows:
        c.q(f"F8.{k}.r1", f"{lab}: R1 (seed 42)", R9 + f"r1_{k}")
        c.q(f"F8.{k}.cp", f"{lab}: C' (seed 42)", R9 + f"c_living_{k}")
        c.q(f"F8.{k}.d", f"{lab}: D (seed 42)", R9 + f"d_{k}")
    for k in ("g2", "g3", "g6"):
        c.q(f"F8.{k}.d", f"pattern {k}: D", R9 + f"d_{k}")
        c.q(f"F8.{k}.cp", f"pattern {k}: C'", R9 + f"c_living_{k}")
    c.q("F8.g4.d", "pattern g4: D", R9 + "d_g4")
    c.q("F8.g4.cp", "pattern g4: C'", R9 + "c_living_g4")
    c.q("F8.g5.d_first", "g5 first run D (priors: stress), seed 42", R9 + "d_g5@_live5")
    c.q("F8.g5.cp_first", "g5 first run C', seed 42", R9 + "c_living_g5@_live5")
    c.q("F8.g5.d_rerun", "g5 rerun D (newer priors), seed 42", R9 + "d_g5@_live6", stat="seed:42")
    c.q(
        "F8.g5.swing", "g5 seed 42 swing from priors file", R9 + "d_g5@_live6", b=R9 + "d_g5@_live5"
    )
    c.q("F8.g5.d3", "g5 D, 3 seeds (newer priors)", R9 + "d_g5")
    c.q("F8.g5.cp3", "g5 C' unmatched, 3 seeds", R9 + "c_living_g5")
    c.q("F8.g5.cm3", "g5 C'-matched (to D-gfix steps), 3 seeds", R9 + "c_living_g5_matched")
    c.q(
        "F8.g5.gap_cm",
        "g5: D (old gate) minus C'-matched",
        R9 + "d_g5",
        b=R9 + "c_living_g5_matched",
    )
    c.q(
        "F8.g5.gap_cm.wins",
        "g5: D wins vs C'-matched (of 3)",
        R9 + "d_g5",
        b=R9 + "c_living_g5_matched",
        stat="wins",
    )
    c.q("F8.g5.steps_d", "g5 D (old gate) steps", R9 + "d_g5", "total_burst_batches")
    c.q(
        "F8.g5.steps_cm",
        "g5 C'-matched steps (matched to gate-fix D)",
        R9 + "c_living_g5_matched",
        "total_burst_batches",
    )
    c.q(
        "F8.g5.gfix_gap",
        "g5: D gate-fix minus C'-matched (equal steps)",
        R9 + "d_g5_gfix",
        b=R9 + "c_living_g5_matched",
        note="equal-step version of the g5 comparison",
    )
    c.q("F8.g5.gfix_d", "g5 D gate-fix", R9 + "d_g5_gfix")
    # Finding 9
    c.q("F9.ew2.old", "told 2 waves, old priors, seed 42", R9 + "d_ew2@_live3", stat="seed:42")
    c.q("F9.correct.old", "told correct, old rate, seed 42", R9 + "d", stat="seed:42")
    c.q("F9.ew2.new3", "told 2 waves, newer priors, all 3 seeds", R9 + "d_ew2")
    c.q("F9.ew2.new2", "told 2 waves, newer priors, seeds 101,102", R9 + "d_ew2", seeds=[101, 102])
    c.q("F9.ew2.new42", "told 2 waves, newer priors, seed 42", R9 + "d_ew2", stat="seed:42")
    c.q("F9.ew6.new3", "told 6 waves, newer priors, all 3 seeds", R9 + "d_ew6")
    c.q("F9.ew6.new2", "told 6 waves, newer priors, seeds 101,102", R9 + "d_ew6", seeds=[101, 102])
    c.q("F9.ew6.new42", "told 6 waves, newer priors, seed 42", R9 + "d_ew6", stat="seed:42")
    c.q("F9.correct.2", "told correct (3 waves), seeds 101,102", D3, seeds=[101, 102])
    c.q("F9.correct.42", "told correct, seed 42", D3, stat="seed:42")
    c.q("F9.ew2.old3", "told 2 waves old priors, 3-seed mean", R9 + "d_ew2@_live3")
    c.q("F9.correct.old3", "D old rate correct, 3-seed mean", R9 + "d")


def finding_10(c: Claims) -> None:
    n = "resnet9_next_deploy_"
    cp = n + "c_living_c100_matched_eta"
    d = n + "d_c100_eta0.3"
    c.q("F10.clean", "CIFAR-100 clean chip, 3 seeds", d, CLEAN_KEY)
    c.q("F10.r1", "R1 CIFAR-100", n + "r1_c100")
    c.q("F10.noburst", "D no bursts CIFAR-100", n + "d_noburst_c100")
    c.q("F10.cp_unmatched", "C' rate 1.0 unmatched", n + "c_living_c100_eta1.0")
    c.q(
        "F10.cp_unmatched.steps",
        "C' unmatched steps",
        n + "c_living_c100_eta1.0",
        "total_burst_batches",
    )
    c.q("F10.cm.0.3", "C'-matched rate 0.3", "resnet9_final_deploy_c_living_c100_matched_eta0.3")
    c.q("F10.cm.0.5", "C'-matched rate 0.5", "resnet9_final_deploy_c_living_c100_matched_eta0.5")
    c.q("F10.cm.1.0", "C'-matched rate 1.0", cp + "1.0")
    c.q("F10.cm.2.0", "C'-matched rate 2.0", cp + "2.0")
    c.q("F10.d", "D rate 0.3", d)
    c.q("F10.d.steps", "D steps", d, "total_burst_batches")
    best = "resnet9_final_deploy_c_living_c100_matched_eta0.5"
    c.q("F10.gap_best", "D minus best C'-matched (0.5)", d, b=best)
    for s in (42, 101, 102):
        c.q(f"F10.gap_best.s{s}", f"D minus best C'-matched seed {s}", d, b=best, stat=f"seed:{s}")
    c.q("F10.gap_best.wins", "D wins vs best C'-matched (of 3)", d, b=best, stat="wins")
    c.q("F10.gap_1.0", "D minus C'-matched rate 1.0", d, b=cp + "1.0")
    c.q("F10.gap_2.0", "D minus C'-matched rate 2.0", d, b=cp + "2.0")
    c.q(
        "F10.keeps",
        "D as % of clean",
        d,
        b=d,
        key="final_test_acc",
        key_b=CLEAN_KEY,
        stat="ratio_pct",
    )
    c.q("F10.cm.steps", "C'-matched steps", best, "total_burst_batches")
    c.q("F10.d_sweep.0.05", "D sweep seed 42 rate 0.05", n + "d_c100_eta0.05")
    c.q("F10.d_sweep.1.0", "D sweep seed 42 rate 1.0", n + "d_c100_eta1.0")
    c.q("F10.old.r1", "first CIFAR-100 test R1 (weak base net)", R9 + "r1_c100")
    c.q("F10.old.d", "first CIFAR-100 test D (weak base net)", R9 + "d_c100")
    c.q(
        "F10.old.clean",
        "first CIFAR-100 test clean chip (weak base net)",
        R9 + "r1_c100",
        CLEAN_KEY,
    )


def finding_11(c: Claims) -> None:
    # learning-rate table, 3 seeds
    old = {
        "B": (R9 + "b", VGG["b_old"], VIT["b_old"]),
        "C": (R9 + "c_living", VGG["c_old"], VIT["c_old"]),
        "D": (R9 + "d", VGG["d_old"], VIT["d_old"]),
    }
    new = {
        "B": (B5, VGG["b3"], VIT["b3"]),
        "C": (C5, VGG["c3"], VIT["c3"]),
        "D": (D3, VGG["d3"], VIT["d3"]),
    }
    nets = ("ResNet-9", "VGG6", "ViT-tiny")
    for arm in "BCD":
        for i, net in enumerate(nets):
            c.q(f"F11.lr.{net}.{arm}.old", f"{net} {arm} old rate 0.005, 3 seeds", old[arm][i])
            c.q(f"F11.lr.{net}.{arm}.new", f"{net} {arm} tuned rate, 3 seeds", new[arm][i])
            c.q(
                f"F11.lr.{net}.{arm}.swing",
                f"{net} {arm} tuned minus old, 3 seeds",
                new[arm][i],
                b=old[arm][i],
            )
    # equal-compute table, 5 seeds
    t = {
        "ResNet-9": {
            "r1": R1_5,
            "bm": BM5,
            "cm": CM5,
            "d": D5,
            "clean": f"{R9}r0+resnet9_final_r1",
        },
        "VGG6": {
            "r1": VGG["r1"],
            "bm": VGG["bm"],
            "cm": VGG["cm"],
            "d": VGG["d"],
            "clean": VGG["r1"],
        },
        "ViT-tiny": {
            "r1": VIT["r1"],
            "bm": VIT["bm"],
            "cm": VIT["cm"],
            "d": VIT["d"],
            "clean": VIT["r1"],
        },
    }
    for net, x in t.items():
        c.q(f"F11.eq.{net}.clean", f"{net} clean, 5 seeds", x["clean"], CLEAN_KEY)
        c.q(
            f"F11.eq.{net}.clean3",
            f"{net} clean, seeds 42/101/102 only",
            x["clean"],
            CLEAN_KEY,
            seeds=[42, 101, 102],
        )
        c.q(f"F11.eq.{net}.r1", f"{net} R1, 5 seeds", x["r1"])
        c.q(f"F11.eq.{net}.bm", f"{net} B-matched, 5 seeds", x["bm"])
        c.q(f"F11.eq.{net}.cm", f"{net} C'-matched, 5 seeds", x["cm"])
        c.q(f"F11.eq.{net}.d", f"{net} D, 5 seeds", x["d"])
        c.q(f"F11.eq.{net}.gap_cm", f"{net} D minus C'-matched", x["d"], b=x["cm"])
        c.q(
            f"F11.eq.{net}.wins_cm",
            f"{net} D wins vs C'-matched (of 5)",
            x["d"],
            b=x["cm"],
            stat="wins",
        )
        c.q(f"F11.eq.{net}.gap_bm", f"{net} D minus B-matched", x["d"], b=x["bm"])
        c.q(
            f"F11.eq.{net}.wins_bm",
            f"{net} D wins vs B-matched (of 5)",
            x["d"],
            b=x["bm"],
            stat="wins",
        )
    c.q("F11.vgg.s101", "VGG6 D minus C'-matched seed 101", VGG["d"], b=VGG["cm"], stat="seed:101")
    c.q("F11.vgg.dmin", "VGG6 D worst seed", VGG["d"], stat="min")
    c.q("F11.vgg.dmax", "VGG6 D best seed", VGG["d"], stat="max")
    c.q("F11.vgg.recal.d", "VGG6 recalibrated D, 3 seeds", "vgg6_net2_deploy_d_recal")
    c.q(
        "F11.vgg.recal.cm",
        "VGG6 recal C'-matched, 3 seeds",
        "vgg6_net2_deploy_c_living_matched_recal",
    )
    c.q(
        "F11.vit.d_over_clean",
        "ViT D as % of clean, mean over 5 seeds",
        VIT["d"],
        b=VIT["d"],
        key_b=CLEAN_KEY,
        stat="ratio_pct",
    )
    for s in (42, 101, 102, 103, 104):
        c.q(
            f"F11.vit.d_over_clean.s{s}",
            f"ViT D seed {s} as % of its clean",
            VIT["d"],
            b=VIT["d"],
            key_b=CLEAN_KEY,
            stat="ratio_pct",
            seeds=[s],
        )
    ratios = [
        100 * r["final_test_acc"] / r["baseline_test_acc"] for r in c.db.resolve(VIT["d"]).values()
    ]
    c.custom("F11.vit.ratio_max", "ViT D as % of clean, best seed", max(ratios), VIT["d"])
    c.custom(
        "F11.vit.ratio_min", "ViT D as % of clean, worst seed (seed 103)", min(ratios), VIT["d"]
    )
    c.q("F11.vit.s103.d", "ViT seed 103 D", VIT["d"], stat="seed:103")
    c.q("F11.vit.s103.clean", "ViT seed 103 clean", VIT["d"], CLEAN_KEY, stat="seed:103")
    c.q("F11.vit.s42.tie", "ViT seed 42: D minus C'-matched", VIT["d"], b=VIT["cm"], stat="seed:42")
    c.q("F11.vgg.s101.d", "VGG6 seed 101 D", VGG["d"], stat="seed:101")


def findings_12_13(c: Claims) -> None:
    for s in ("s1", "s2", "s3"):
        p = f"resnet9_real_{s}_"
        c.q(f"F12.{s}.r1", f"{s} R1", p + "r1")
        c.q(f"F12.{s}.bm", f"{s} B-matched", p + "b_matched")
        c.q(f"F12.{s}.cm", f"{s} C'-matched", p + "c_living_matched")
        c.q(f"F12.{s}.d", f"{s} D", p + "d")
        c.q(f"F12.{s}.gap", f"{s} D minus C'-matched", p + "d", b=p + "c_living_matched")
        c.q(
            f"F12.{s}.wins",
            f"{s} D wins vs C'-matched (of 3)",
            p + "d",
            b=p + "c_living_matched",
            stat="wins",
        )
        c.q(f"F12.{s}.gap_bm", f"{s} D minus B-matched", p + "d", b=p + "b_matched")
        c.q(f"F12.{s}.steps_d", f"{s} D steps", p + "d", "total_burst_batches")
        c.q(
            f"F12.{s}.steps_cm",
            f"{s} C'-matched steps",
            p + "c_living_matched",
            "total_burst_batches",
        )
    lr = (
        ("0.005", "lr0.005"),
        ("0.05", "lr0.05"),
        ("0.1", "lr0.1"),
        ("0.3", "lr0.3"),
        ("0.5", "lr0.5"),
    )
    for lab, suf in lr:
        c.q(f"F13.liu.{lab}", f"Liu-style at burst rate {lab}", R9 + f"liu_style_{suf}")
    c.q(
        "F13.swing",
        "Liu-style rate 0.5 minus 0.005",
        R9 + "liu_style_lr0.5",
        b=R9 + "liu_style_lr0.005",
    )
    c.q("F13.cm", "C'-matched 3 seeds", CM)
    c.q("F13.d", "D 3 seeds", D3)


def finding_14(c: Claims) -> None:
    for k in ("s4a", "s4b", "s4a5", "s4b5"):
        p = f"resnet9_calib_{k}_"
        c.q(f"F14.{k}.r1", f"{k} R1", p + "r1")
        c.q(f"F14.{k}.bm", f"{k} B-matched", p + "b_matched")
        c.q(f"F14.{k}.d", f"{k} D", p + "d")
        c.q(f"F14.{k}.gap_bm", f"{k} D minus B-matched", p + "d", b=p + "b_matched")
        c.q(
            f"F14.{k}.wins_bm",
            f"{k} D wins vs B-matched (of 3)",
            p + "d",
            b=p + "b_matched",
            stat="wins",
        )
        c.q(f"F14.{k}.steps", f"{k} D steps", p + "d", "total_burst_batches")
        c.q(f"F14.{k}.steps_bm", f"{k} B-matched steps", p + "b_matched", "total_burst_batches")
    p = "resnet9_calib_s4b5_"
    c.q("F14.s4b5.cm", "S4b5 C'-matched", p + "c_living_matched")
    c.q("F14.s4b5.gap_cm", "S4b5 D minus C'-matched", p + "d", b=p + "c_living_matched")
    c.q(
        "F14.s4b5.wins_cm",
        "S4b5 D wins vs C'-matched (of 3)",
        p + "d",
        b=p + "c_living_matched",
        stat="wins",
    )
    c.q("F14.s4b5.steps_cm", "S4b5 C'-matched steps", p + "c_living_matched", "total_burst_batches")
    c.q(
        "F14.bm_better_min",
        "smallest B-matched advantage over D",
        "resnet9_calib_s4a_b_matched",
        b="resnet9_calib_s4a_d",
    )
    c.q(
        "F14.bm_better_max",
        "largest B-matched advantage over D",
        "resnet9_calib_s4b5_b_matched",
        b="resnet9_calib_s4b5_d",
    )


def finding_15(c: Claims) -> None:
    for k in ("s4a", "s4b", "s4a5", "s4b5"):
        o, p = f"resnet9_calib_{k}_d", f"resnet9_d2calib_{k}_"
        c.q(f"F15.{k}.d", f"{k} D (old)", o)
        c.q(f"F15.{k}.d2", f"{k} D2", p + "d2")
        c.q(f"F15.{k}.bm", f"{k} B-matched (D2 steps)", p + "b_matched")
        c.q(f"F15.{k}.c2", f"{k} C2", p + "c2")
        c.q(f"F15.{k}.d2_d", f"{k} D2 minus D", p + "d2", b=o)
        c.q(f"F15.{k}.d2_bm", f"{k} D2 minus B-matched", p + "d2", b=p + "b_matched")
        c.q(f"F15.{k}.d2_c2", f"{k} D2 minus C2", p + "d2", b=p + "c2")
        c.q(f"F15.{k}.steps_d2", f"{k} D2 steps", p + "d2", "total_burst_batches")
        c.q(f"F15.{k}.steps_d", f"{k} D (old) steps", o, "total_burst_batches")
    c.q("F15.s4b5.cm", "S4b5 C'-matched (D2 steps)", "resnet9_d2calib_s4b5_c_living_matched")
    c.q(
        "F15.s4b5.cm_minus_d2",
        "S4b5 C'-matched minus D2",
        "resnet9_d2calib_s4b5_c_living_matched",
        b="resnet9_d2calib_s4b5_d2",
    )
    m = "resnet9_d2_"
    d5 = D5
    c.q("F15.main.d", "main test D (old), 5 seeds", d5)
    c.q("F15.main.d.steps", "main test D steps", d5, "total_burst_batches")
    c.q("F15.main.d2", "main test D2, 5 seeds", m + "d2")
    c.q("F15.main.d2.steps", "D2 steps", m + "d2", "total_burst_batches")
    c.q("F15.main.cm", "C'-matched (D2 steps)", m + "c_living_matched")
    c.q("F15.main.cm.steps", "C'-matched steps", m + "c_living_matched", "total_burst_batches")
    c.q("F15.main.bm", "B-matched (D2 steps)", m + "b_matched")
    c.q("F15.main.bm.steps", "B-matched steps", m + "b_matched", "total_burst_batches")
    c.q("F15.main.c2", "C2", m + "c2")
    c.q("F15.main.c2.steps", "C2 steps", m + "c2", "total_burst_batches")
    c.q("F15.main.d2_d", "D2 minus D", m + "d2", b=d5)
    for s in (42, 101, 102, 103, 104):
        c.q(f"F15.main.d2_d.s{s}", f"D2 minus D seed {s}", m + "d2", b=d5, stat=f"seed:{s}")
    c.q("F15.main.d2_cm", "D2 minus C'-matched", m + "d2", b=m + "c_living_matched")
    c.q(
        "F15.main.d2_cm.wins",
        "D2 wins vs C'-matched (of 5)",
        m + "d2",
        b=m + "c_living_matched",
        stat="wins",
    )
    c.q("F15.main.d2_bm", "D2 minus B-matched", m + "d2", b=m + "b_matched")
    c.q("F15.main.d2_c2", "D2 minus C2", m + "d2", b=m + "c2")
    c.q("F15.p2.d2_vs_limit", "P2: D2 minus D must be >= -0.3", m + "d2", b=d5)
    # C2 within 0.6 of D2 on the four S4 scenarios
    worst = max(
        abs(_mean_diff(c.db, f"resnet9_d2calib_{k}_c2", f"resnet9_d2calib_{k}_d2"))
        for k in ("s4a", "s4b", "s4a5", "s4b5")
    )
    c.custom(
        "F15.c2_within",
        "max |C2 - D2| over four S4 scenarios",
        worst,
        "resnet9_d2calib_*_c2 vs *_d2",
    )
    c.q(
        "F15.d2_gain_min",
        "min D2-D gain over S4",
        "resnet9_d2calib_s4a_d2",
        b="resnet9_calib_s4a_d",
    )
    c.q(
        "F15.d2_gain_max",
        "max D2-D gain over S4",
        "resnet9_d2calib_s4b5_d2",
        b="resnet9_calib_s4b5_d",
    )


def sa_classic_claims(c: Claims) -> None:
    """The older classic arms (resnet9_fault_mixed_*)."""
    classic = (
        "no_mitigation",
        "soft_mitigation",
        "layer_reset",
        "remapping",
        "remapping_rowcol",
        "combined",
        "combined_rowcol",
        "remapping_rf05",
        "remapping_rf10",
        "remapping_rf20",
        "remapping_rf30",
    )
    for arm in classic:
        c.q(
            f"SA3.{arm}.last",
            f"classic arm {arm}: last logged test acc (batch 19400)",
            f"resnet9_fault_mixed_{arm}",
            "metrics_last_acc",
        )
        c.q(
            f"SA3.{arm}.mean5",
            f"classic arm {arm}: mean of last 5 logs",
            f"resnet9_fault_mixed_{arm}",
            "metrics_last5_mean",
        )


def _first_config(run_dir_name: str) -> dict:
    for p in sorted(RESULTS.glob(f"_live*/{run_dir_name}/config.json")):
        return _load(p) or {}
    return {}


def _mean_diff(db: DB, a: str, b: str) -> float:
    ra, rb = db.resolve(a), db.resolve(b)
    common = sorted(set(ra) & set(rb))
    return st.mean(ra[s]["final_test_acc"] - rb[s]["final_test_acc"] for s in common)


def section_5(c: Claims) -> None:
    # section 5 point 3: distance of each arm to clean (ResNet-9, 3 seeds)
    cl = st.mean(r["baseline_test_acc"] for r in c.db.resolve(R9 + "r0").values())
    for lab, arm in (("D", D3), ("C'-matched", CM), ("B-matched", BM), ("B tuned", B5)):
        vals = [r["final_test_acc"] for r in c.db.resolve(arm).values()]
        c.custom(
            f"S5.3.gap_to_clean.{lab}", f"clean minus {lab} (3 seeds)", cl - st.mean(vals), arm
        )
    gaps_clean = [
        cl - st.mean(r["final_test_acc"] for r in c.db.resolve(a).values())
        for a in (D3, CM, BM, B5)
    ]
    c.custom(
        "S5.3.gap_min",
        "smallest gap to clean over D, C'M, BM, B",
        min(gaps_clean),
        "S5.3.gap_to_clean.*",
    )
    c.custom(
        "S5.3.gap_max",
        "largest gap to clean over D, C'M, BM, B",
        max(gaps_clean),
        "S5.3.gap_to_clean.*",
    )
    # point 4: counts of ResNet-9 cases where D wins all seeds against C'-matched
    cases = [(D5, CM5, 5)]
    cases += [
        (R9 + f"d_stress_{k}", R9 + f"c_living_stressM_{k}", 3)
        for k in ("rf02", "rf05", "rf10", "d15", "d20", "d25")
    ]
    cases += [
        (f"resnet9_real_{s}_d", f"resnet9_real_{s}_c_living_matched", 3) for s in ("s1", "s2", "s3")
    ]
    full = 0
    for a, b, n in cases:
        ra, rb = c.db.resolve(a), c.db.resolve(b)
        w = sum(ra[s]["final_test_acc"] > rb[s]["final_test_acc"] for s in set(ra) & set(rb))
        full += int(w == n)
    c.custom(
        "S5.4.cases_total",
        "ResNet-9 cases counted (main + 6 stress + 3 realistic)",
        len(cases),
        "list above",
    )
    c.custom(
        "S5.4.cases_all_wins", "cases where D wins every seed vs C'-matched", full, "list above"
    )
    # point 5: D minus B-matched across ResNet-9 scenarios
    gaps = {}
    for k in ("rf02", "rf05", "rf10", "d15", "d20", "d25"):
        gaps[f"stress_{k}"] = _mean_diff(c.db, R9 + f"d_stress_{k}", R9 + f"b_stressM_{k}")
    for s in ("s1", "s2", "s3"):
        gaps[f"real_{s}"] = _mean_diff(c.db, f"resnet9_real_{s}_d", f"resnet9_real_{s}_b_matched")
    for k in ("s4a", "s4b", "s4a5", "s4b5"):
        gaps[f"calib_{k}"] = _mean_diff(
            c.db, f"resnet9_calib_{k}_d", f"resnet9_calib_{k}_b_matched"
        )
    c.custom(
        "S5.5.min_all",
        "min D minus B-matched over stress+realistic+S4",
        min(gaps.values()),
        ";".join(f"{k}={v:+.2f}" for k, v in gaps.items()),
        note="S4 cases are negative: claim 'all ResNet-9 scenarios' is false",
    )
    keep = {k: v for k, v in gaps.items() if not k.startswith("calib")}
    c.custom(
        "S5.5.min_no_s4",
        "min D minus B-matched over stress+realistic (no S4)",
        min(keep.values()),
        "stress+real",
    )
    c.custom(
        "S5.5.max_no_s4",
        "max D minus B-matched over stress+realistic (no S4)",
        max(keep.values()),
        "stress+real",
    )
    c.custom(
        "S5.5.n_s4_losses",
        "S4 scenarios where D is below B-matched",
        sum(v < 0 for k, v in gaps.items() if k.startswith("calib")),
        "calib",
    )
    # point 2: rate effect range
    swings = []
    for b_old, b_new, cp_old, cp_new in (
        (R9 + "b", B5, R9 + "c_living", C5),
        (VGG["b_old"], VGG["b3"], VGG["c_old"], VGG["c3"]),
        (VIT["b_old"], VIT["b3"], VIT["c_old"], VIT["c3"]),
    ):
        swings += [_mean_diff(c.db, b_new, b_old), _mean_diff(c.db, cp_new, cp_old)]
    c.custom(
        "S5.2.swing_min",
        "smallest tuned-minus-old gain for B and C' over 3 nets",
        min(swings),
        "F11 lr table",
    )
    c.custom(
        "S5.2.swing_max",
        "largest tuned-minus-old gain for B and C' over 3 nets",
        max(swings),
        "F11 lr table",
    )
    c.q("S5.6.vit", "ViT D minus C'-matched (5 seeds)", VIT["d"], b=VIT["cm"])
    c.q("S5.6.vgg", "VGG6 D minus C'-matched (5 seeds)", VGG["d"], b=VGG["cm"])
    # same-config reruns: largest per-seed difference between run BD2 and BD
    noise = []
    for stem in ("d_nopace", "d_nosal", "d_mag"):
        a, b = c.db.resolve(R9 + stem + "_etaBD2"), c.db.resolve(R9 + stem + "_etaBD")
        noise += [abs(a[s]["final_test_acc"] - b[s]["final_test_acc"]) for s in set(a) & set(b)]
    c.custom(
        "S5.noise.floor",
        "largest same-config rerun per-seed difference",
        max(noise),
        "resnet9_deploy_d_{nopace,nosal,mag}_etaBD vs etaBD2",
    )


def config_facts(c: Claims) -> None:
    accs = []
    for p in sorted(RESULTS.glob("resnet9_ibm_recipe_hwa_full_calibrated_gpu_seed*")):
        m = _load(p / "metrics.json")
        if m and m.get("accuracy"):
            accs.append(m["accuracy"][-1])
    c.custom(
        "intro.clean.pretrain50",
        "50-epoch fault-free ResNet-9, last logged test acc, mean",
        st.mean(accs) if accs else None,
        f"{len(accs)} results/resnet9_ibm_recipe_* dirs",
        note="MANIFEST says 89.79 (4 seeds); different, longer-trained net than the "
        "14-epoch deployment base net (89.0)",
    )
    cfg = _load(RESULTS / "_live2" / "resnet9_deploy_d_seed42" / "config.json") or {}
    test_n = cfg.get("test_samples")
    c.custom("cfg.test_images", "test images per evaluation", test_n, "config.json")
    c.custom(
        "cfg.canary", "canary size", cfg.get("retraining", {}).get("canary_size"), "config.json"
    )
    c.custom(
        "cfg.spares",
        "spare cell fraction",
        cfg.get("crossbar", {}).get("redundancy_factor"),
        "config.json",
    )
    c.custom(
        "cfg.waves",
        "default wave densities",
        sum(cfg.get("fault", {}).get("wave_densities") or []),
        "config.json",
    )
    for s in ("s1", "s2", "s3"):
        cf = _load(RESULTS / "_live9" / f"resnet9_real_{s}_d_seed42" / "config.json") or {}
        dens = (cf.get("fault") or {}).get("wave_densities") or []
        prod = 1.0
        for x in dens:
            prod *= 1 - x
        c.custom(
            f"cfg.real.{s}.stuck",
            f"{s} cumulative stuck fraction 1-prod(1-d)",
            1 - prod,
            "config.json wave_densities",
        )
    for k in ("s4a", "s4b"):
        cf = _load(RESULTS / "_live13" / f"resnet9_calib_{k}_d_seed42" / "config.json") or {}
        dens = (cf.get("fault") or {}).get("wave_densities") or []
        prod = 1.0
        for x in dens:
            prod *= 1 - x
        c.custom(
            f"cfg.calib.{k}.stuck",
            f"{k} cumulative stuck fraction",
            1 - prod,
            "config.json wave_densities",
        )
        c.custom(
            f"cfg.calib.{k}.maxwave",
            f"{k} largest wave density",
            max(dens) if dens else None,
            "config.json",
        )
    cf = _load(RESULTS / "_live13" / "resnet9_calib_s4a_d_seed42" / "config.json") or {}
    c.custom(
        "cfg.calib.lrs",
        "fraction of new stuck cells stuck-at-LRS",
        (cf.get("fault") or {}).get("stuck_lrs_fraction"),
        "config.json",
    )
    sec = cf.get("secondary_fault") or {}
    c.custom(
        "cfg.calib.drift",
        "drift per wave (fraction of conductance range)",
        sec.get("drift_magnitude"),
        "config.json",
    )
    c.custom(
        "cfg.calib.weibull",
        "Weibull shape",
        (cf.get("fault") or {}).get("weibull_shape"),
        "config.json",
    )

    # unremapped cells in R1 (static remap, uses all spares): C' only retrains when this is > 0
    for k in ("s4a", "s4b", "s4a5", "s4b5"):
        c.q(
            f"F14.{k}.unremapped",
            f"{k}: max unremapped stuck cells over waves (R1)",
            f"resnet9_calib_{k}_r1",
            "max_unremapped",
            stat="max",
            note="0 means the spare cells never ran out (so C' never retrains)",
        )
    # D2 plateau minimum gain
    c.q(
        "F15.mingain.d2",
        "D2 mean plateau_min_gain over waves, main test",
        "resnet9_d2_d2",
        "plateau_min_gain_mean",
    )
    # model facts not stored in results
    c.custom(
        "cfg.vit.params",
        "ViT-tiny weights (millions)",
        None,
        "no param count in results",
        note="check src/neurofault/models/cnn.py by hand",
    )
    c.custom(
        "cfg.vit.blocks",
        "ViT-tiny blocks",
        None,
        "not in results config.json",
        note="src/neurofault/models/cnn.py:216 has depth=4",
    )
    cc100 = _first_config("resnet9_next_deploy_r1_c100_seed42")
    c.custom("cfg.c100.epochs", "CIFAR-100 base network epochs", cc100.get("epochs"), "config.json")
    c.custom(
        "cfg.main.epochs",
        "continuous regime training epochs",
        (_first_config("resnet9_cont_next_d_seed101")).get("epochs"),
        "config.json",
    )
    # stress / pattern definitions (number of waves and wave sizes)
    for k in ("g2", "g3", "g4", "g5", "g1"):
        wd = (_first_config(f"resnet9_deploy_d_{k}_seed42").get("fault") or {}).get(
            "wave_densities"
        ) or []
        c.custom(f"cfg.{k}.nwaves", f"pattern {k}: number of waves", len(wd), "config.json")
        c.custom(
            f"cfg.{k}.density",
            f"pattern {k}: wave density (mean)",
            st.mean(wd) if wd else None,
            "config.json",
        )
    wd = (_first_config("resnet9_deploy_d_g6_seed42").get("fault") or {}).get("wave_densities")
    c.custom(
        "cfg.g6.sum",
        "pattern g6: sum of wave densities (15%+5%+10%)",
        sum(wd) if wd else None,
        "config.json",
    )
    for k in ("w6", "w8"):
        wd = (_first_config(f"resnet9_deploy_d_stress_{k}_seed42").get("fault") or {}).get(
            "wave_densities"
        ) or []
        c.custom(f"cfg.stress.{k}.nwaves", f"stress {k}: number of waves", len(wd), "config.json")
    n_cases = sum(
        bool(c.db.resolve(R9 + f"d_stress_{k}"))
        and bool(c.db.resolve(R9 + f"c_living_stressM_{k}"))
        for k in ("rf02", "rf05", "rf10", "d15", "d20", "d25")
    )
    c.custom(
        "cfg.stress.n_non_wave_cases",
        "stress cases (not 'small waves') with D and C'-matched runs",
        n_cases,
        "run names rf02..d25",
        note="six rows: 2%, 5%, 10% spares; 15%, 20%, 25% faults per wave",
    )
    # D2 design constants
    d2 = _first_config("resnet9_d2_d2_seed42").get("retraining") or {}
    c.custom(
        "cfg.d2.k",
        "D2 gate threshold in canary standard errors",
        d2.get("gate_noise_k"),
        "config.json",
    )
    c.custom("cfg.d2.anneal", "D2 anneal steps", d2.get("anneal_steps"), "config.json")
    # P1 counts
    d2_vs_b = [
        _mean_diff(c.db, f"resnet9_d2calib_{k}_d2", f"resnet9_d2calib_{k}_b_matched")
        for k in ("s4a", "s4b", "s4a5", "s4b5")
    ]
    d_vs_b = [
        _mean_diff(c.db, f"resnet9_calib_{k}_d", f"resnet9_calib_{k}_b_matched")
        for k in ("s4a", "s4b", "s4a5", "s4b5")
    ]
    c.custom(
        "F15.p1.count",
        "S4 scenarios where D2 is within 0.5 of B-matched (>= -0.5)",
        sum(x >= -0.5 for x in d2_vs_b),
        "d2calib vs b_matched",
        note="pre-registered rule P1 needs at least 3 of 4",
    )
    c.custom(
        "F15.p1.count_d",
        "S4 scenarios where original D is behind B-matched",
        sum(x < 0 for x in d_vs_b),
        "calib d vs b_matched",
    )
    # flatness of the seed-42 test curve between rates 0.2 and 0.5
    for arm, stem in (("B", R9 + "b_eta"), ("C-prime", R9 + "c_living_eta"), ("D", R9 + "d_eta")):
        vals = [
            c.db.resolve(stem + suf).get(42, {}).get("final_test_acc")
            for suf in ("2e-1", "3e-1", "5e-1")
        ]
        vals = [v for v in vals if v is not None]
        c.custom(
            f"RR.flat.{arm}",
            f"{arm} seed 42: max-min test acc over rates 0.2-0.5",
            max(vals) - min(vals),
            "seed-42 sweep",
        )


def prereg_claims(c: Claims) -> None:
    """Numbers used in the D2 adaptive-trigger pre-registration."""
    for k in ("s4a", "s4b", "s4a5", "s4b5"):
        p = f"resnet9_calib_{k}_"
        c.q(f"PR.orig_d.{k}", f"original D minus B-matched, {k}", p + "d", b=p + "b_matched")
    c.q("PR.p2.d", "D mean (main test, 5 seeds)", D5)
    c.q("PR.p3.d", "D mean VGG6 (5 seeds)", VGG["d"])
    c.q("PR.p3.worst", "worst D seed VGG6", VGG["d"], stat="min")
    c.custom(
        "PR.sigma",
        "canary sigma at p=0.9, n=1024 (points)",
        100 * (0.9 * 0.1 / 1024) ** 0.5,
        "arithmetic",
    )
    c.q("PR.out.d2", "stage 1: D2 main test mean", "resnet9_d2_d2")
    c.q(
        "PR.out.d2_cm",
        "stage 1: D2 minus C'-matched",
        "resnet9_d2_d2",
        b="resnet9_d2_c_living_matched",
    )


def canary_claims(c: Claims) -> None:
    """Learning-rate picks and 'test lost' numbers of the canary-selection check."""
    rows = canary_rows(c.db)
    lost3: list[float] = []
    for arm in ("B", "C-prime", "D"):
        mine = [r for r in rows if r["arm"] == arm and r["s42_final_test_acc"]]
        test = {float(r["lr"]): float(r["s42_final_test_acc"]) for r in mine}
        last = {float(r["lr"]): float(r["s42_canary_last"]) for r in mine}
        mean = {float(r["lr"]): float(r["s42_canary_mean"]) for r in mine}
        best = max(test, key=test.get)
        pick_l, pick_m = max(last, key=last.get), max(mean, key=mean.get)
        c.custom(f"CC.s42.{arm}.best_lr", f"{arm} seed 42 best LR by test", best, "sweep")
        c.custom(f"CC.s42.{arm}.best_test", f"{arm} seed 42 best test acc", test[best], "sweep")
        c.custom(
            f"CC.s42.{arm}.pick_last", f"{arm} seed 42 canary(last wave) pick", pick_l, "sweep"
        )
        c.custom(
            f"CC.s42.{arm}.lost_last",
            f"{arm} seed 42 test lost, canary(last) pick",
            test[best] - test[pick_l],
            "sweep",
        )
        c.custom(f"CC.s42.{arm}.pick_mean", f"{arm} seed 42 canary(mean) pick", pick_m, "sweep")
        c.custom(
            f"CC.s42.{arm}.lost_mean",
            f"{arm} seed 42 test lost, canary(mean) pick",
            test[best] - test[pick_m],
            "sweep",
        )
        three = [r for r in rows if r["arm"] == arm and r["mean3_final_test_acc"]]
        t3 = {float(r["lr"]): float(r["mean3_final_test_acc"]) for r in three}
        l3 = {float(r["lr"]): float(r["mean3_canary_last"]) for r in three}
        m3 = {float(r["lr"]): float(r["mean3_canary_mean"]) for r in three}
        b3 = max(t3, key=t3.get)
        pl, pm = max(l3, key=l3.get), max(m3, key=m3.get)
        lost3 += [t3[b3] - t3[pl], t3[b3] - t3[pm]]
        c.custom(f"CC.s3.{arm}.best_lr", f"{arm} 3-seed best LR by test", b3, "sweep")
        c.custom(f"CC.s3.{arm}.pick_last", f"{arm} 3-seed canary(last) pick", pl, "sweep")
        c.custom(
            f"CC.s3.{arm}.lost_last",
            f"{arm} 3-seed test lost, canary(last) pick",
            t3[b3] - t3[pl],
            "sweep",
        )
        c.custom(f"CC.s3.{arm}.pick_mean", f"{arm} 3-seed canary(mean) pick", pm, "sweep")
        c.custom(
            f"CC.s3.{arm}.lost_mean",
            f"{arm} 3-seed test lost, canary(mean) pick",
            t3[b3] - t3[pm],
            "sweep",
        )
    c.custom(
        "CC.max_lost3", "largest 3-seed test loss from canary pick", max(lost3), "see CC.s3.* rows"
    )


# --------------------------------------------------------------------------
# Part 3: canary-based learning-rate selection (canary-selection check)
# --------------------------------------------------------------------------
LR_NAMES = {
    "0.001": "1e-3",
    "0.002": "2e-3",
    "0.01": "1e-2",
    "0.02": "2e-2",
    "0.05": "5e-2",
    "0.1": "1e-1",
    "0.2": "2e-1",
    "0.3": "3e-1",
    "0.5": "5e-1",
    "1": "1e0",
    "2": "2",
}


def canary_rows(db: DB) -> list[dict]:
    out = []
    for arm, stem in (("B", R9 + "b_eta"), ("C-prime", R9 + "c_living_eta"), ("D", R9 + "d_eta")):
        for lr, suf in LR_NAMES.items():
            runs = db.resolve(stem + suf)
            if not runs:
                continue
            row = {"arm": arm, "lr": lr, "run": stem + suf, "seeds": " ".join(map(str, runs))}
            for key in ("final_test_acc", "canary_last", "canary_mean"):
                row[f"s42_{key}"] = _fmt(runs[42][key]) if 42 in runs else ""
                if len(runs) >= 3:
                    row[f"mean3_{key}"] = _fmt(st.mean(r[key] for r in runs.values()))
                else:
                    row[f"mean3_{key}"] = ""
            out.append(row)
    return out


# --------------------------------------------------------------------------
def _write(path: Path, rows: list[dict], first: list[str] | None = None) -> None:
    cols: list[str] = list(first or [])
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canon, allrows = collect()
    db = DB(canon, allrows)
    first = [
        "run_name",
        "seed",
        "source_dir",
        "complete",
        "duplicate_flag",
        "n_copies",
        "regime",
        "policy",
        "final_test_acc",
        "end_of_run_test_acc",
        "eor_any",
        "eor_source",
        "lifetime_mean",
        "worst_dip",
        "total_burst_batches",
        "total_optimizer_steps",
        "baseline_test_acc",
        "checkpoint_config_hash",
    ]
    _write(OUT / "canonical_runs.csv", canon, first)
    claims = build_claims(db)
    cfirst = [
        "claim_id",
        "description",
        "arms",
        "measure",
        "stat",
        "seeds",
        "per_seed",
        "mean",
        "sd",
        "paired_diffs",
        "n_wins",
        "derived_value",
        "same_checkpoint",
        "source_runs",
    ]
    _write(OUT / "claims.csv", claims, cfirst)
    _write(OUT / "canary_selection.csv", canary_rows(db))
    dups = [r for r in canon if r["duplicate_flag"] == "DIFFERENT"]
    print(
        f"{len(allrows)} run dirs, {len(canon)} canonical runs, "
        f"{len(dups)} duplicates with different content, {len(claims)} claims"
    )
    n_missing = sum(r.get("derived_value") == "NO-SOURCE" for r in claims)
    print(f"  claims without a source run: {n_missing}")


if __name__ == "__main__":
    main()

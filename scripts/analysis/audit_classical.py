"""Read-only audit of vertical 1 (experimental setup, classical static mitigation, fidelity bugs).

Re-derives every number the project documentation quotes about
(a) the fault-free baselines, (b) the moderate-regime classical comparison, (c) the post-fix
classical re-runs (resnet9_fault_*, resnet9_fault_mixed_*, cont_next_classic_*) and the
deployment-regime classical-like arms (R0, R1, B, C), directly from raw run directories, and
checks each against which bug fixes the run predates.

Never writes into raw run directories.  Writes only

    results/derived/classical_claims.csv

and prints tables to stdout (the audit document quotes them).  Optional CPU probes (flags):

    --probe-legacy-dac   A/B of calibrate_dac on the ORIGINAL Sep-11 small CNN (fault-free, 3 epochs)
    --probe-clipping     fraction of layer inputs outside [-1, 1] for a trained ResNet-9 checkpoint

Usage: uv run python scripts/analysis/audit_classical.py [--probe-legacy-dac] [--probe-clipping]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "results" / "derived"
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


def _ist(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=IST).astimezone(UTC)


# Commit times (git author date, converted to UTC).  The working tree usually held a change
# EARLIER than its commit, so "run ended before commit" means "possibly pre-fix" and is only
# decisive together with a config flag or a counter in the run itself (see exposure()).
T_FRAMEWORK = _ist(2026, 9, 14, 14, 23)  # 0e873e8: sync, persistence, dispatch precision, ...
T_ADC = _ist(2026, 9, 24, 17, 46)  # cebb7c2: calibrate_adc, float pretrain
T_DAC = _ist(2026, 9, 28, 21, 27)  # 85e2c59: calibrate_dac
T_REMAP_FIX = _ist(2026, 10, 2, 22, 56)  # 485b54e: remap bugs 1 and 2, seeded generator
T_FREEZE = _ist(2026, 10, 3, 0, 46)  # e58d312: stuck-weight freeze, living spares
T_PREPROV = _ist(2026, 9, 28, 21, 27)  # 85e2c59: run.log/run_metadata provenance appears
T_GEN_CLU = _ist(2026, 10, 8, 17, 28)  # 42ed9a5: clustered-selection + CUDA generator fixes

SEED_RE = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)$")
EVAL_CHANCE = 10.0


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _ts(line: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


@dataclass
class Run:
    name: str
    seed: int
    path: Path
    cfg: dict
    metrics: dict | None
    events: dict | None
    meta: dict | None
    start: datetime | None = None
    end: datetime | None = None
    time_source: str = ""
    n_copies: int = 1
    copies_agree: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO))

    @property
    def fpe(self) -> int:
        return int(self.cfg.get("float_pretrain_epochs") or 0)

    def acc(self) -> list[float]:
        return [float(x) for x in (self.metrics or {}).get("accuracy", [])]

    def analog_idx(self) -> list[int]:
        m = self.metrics or {}
        ep = m.get("epoch", [])
        return [i for i, e in enumerate(ep) if e >= self.fpe]


def _times(path: Path, cfg_dir_is_top: bool):
    log = path / "run.log"
    if log.is_file():
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        ts = [t for t in (_ts(ln) for ln in lines) if t]
        if ts:
            return ts[0], ts[-1], "run.log"
    return None, None, "none"


def index_runs() -> dict[tuple[str, int], Run]:
    """Canonical Run per (name, seed): top-level preferred, else newest _live* copy."""
    cands: dict[tuple[str, int], list[Run]] = {}
    dirs: list[tuple[Path, bool]] = []
    for top in sorted(RESULTS.iterdir()):
        if not top.is_dir() or "spectral" in top.name or top.name.startswith("_logs"):
            continue
        if top.name.startswith("_live"):
            dirs += [(d, False) for d in sorted(top.iterdir()) if d.is_dir()]
        else:
            dirs.append((top, True))
    for d, is_top in dirs:
        cfg = _load(d / "config.json")
        if not isinstance(cfg, dict) or not (d / "metrics.json").is_file():
            continue
        mt = SEED_RE.match(d.name)
        name = mt["name"] if mt else d.name
        seed = int(mt["seed"]) if mt else int(cfg.get("seed", -1))
        run = Run(
            name=name,
            seed=seed,
            path=d,
            cfg=cfg,
            metrics=_load(d / "metrics.json"),
            events=_load(d / "events.json"),
            meta=_load(d / "run_metadata.json"),
        )
        run.start, run.end, run.time_source = _times(d, is_top)
        cands.setdefault((name, seed), []).append(run)
    out: dict[tuple[str, int], Run] = {}
    for key, rs in cands.items():
        rs.sort(
            key=lambda r: (
                (r.path / "run.log").is_file(),
                r.path.parts[-2] == "results",
                r.path.stat().st_mtime,
            ),
            reverse=True,
        )
        best = rs[0]
        best.n_copies = len(rs)
        best.copies_agree = all(r.acc() == best.acc() for r in rs)
        out[key] = best
    return out


IDX = index_runs()


def R(name: str, seeds=None) -> list[Run]:
    rs = [r for (n, s), r in sorted(IDX.items()) if n == name and (seeds is None or s in seeds)]
    return rs


def RR(*names: str) -> list[Run]:
    out: list[Run] = []
    for n in names:
        out += R(n)
    return sorted(out, key=lambda r: r.seed)


# ----------------------------------------------------------------------------------------
# measures
# ----------------------------------------------------------------------------------------
def measure(run: Run, key: str) -> float | None:
    acc = run.acc()
    m = run.metrics or {}
    ev = run.events or {}
    s = ev.get("summary") or {}
    ai = run.analog_idx()
    if key == "final":
        return acc[-1] if acc else None
    if key == "max":
        return max(acc) if acc else None
    if key == "max_analog":
        return max(acc[i] for i in ai) if ai else None
    if key == "min_analog":
        return min(acc[i] for i in ai) if ai else None
    if key == "last5":
        return sum(acc[-5:]) / len(acc[-5:]) if acc else None
    if key == "first_analog":
        return acc[ai[0]] if ai else None
    if key == "last_pretrain":
        return acc[ai[0] - 1] if ai and ai[0] > 0 else None
    if key == "analog_mean":
        return sum(acc[i] for i in ai) / len(ai) if ai else None
    if key == "peak_pre_fault":
        return max(acc[: max(1, len(acc) // 3)])
    if key in ("soft", "remap", "resets", "faults", "health"):
        col = {
            "soft": "soft_mitigations",
            "remap": "remappings",
            "resets": "layer_resets",
            "faults": "faults",
            "health": "health",
        }[key]
        return float(m[col][-1]) if m.get(col) else None
    if key == "max_jump":
        return max(abs(b - a) for a, b in itertools.pairwise(acc)) if len(acc) > 1 else None
    if key == "ev_final":
        return s.get("final_test_acc")
    if key == "ev_base":
        return ev.get("baseline_test_acc")
    if key == "ev_lifetime":
        return s.get("lifetime_mean_test_acc")
    if key == "ev_burst":
        return s.get("total_burst_batches")
    if key == "eor":  # end-of-run: summary value if present, else last metrics point
        v = s.get("end_of_run_test_acc")
        return v if v is not None else (acc[-1] if acc else None)
    if key == "wave3":  # accuracy right after the last wave, continuous regime
        v = s.get("final_test_acc")
        return v if v is not None else (acc[-1] if acc else None)
    raise KeyError(key)


def vals(runs: list[Run], key: str) -> list[float]:
    return [v for v in (measure(r, key) for r in runs) if v is not None]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs, ddof=1):
    return (
        statistics.stdev(xs)
        if len(xs) > 1 and ddof == 1
        else (statistics.pstdev(xs) if len(xs) > 1 else None)
    )


def paired(a: list[Run], b: list[Run], key: str, key_b: str | None = None):
    """A minus B, paired by seed over the seeds present in both."""
    ka = {r.seed: measure(r, key) for r in a}
    kb = {r.seed: measure(r, key_b or key) for r in b}
    seeds = sorted(set(ka) & set(kb))
    d = [ka[s] - kb[s] for s in seeds if ka[s] is not None and kb[s] is not None]
    n = len(d)
    res = {"seeds": seeds, "diffs": d, "n": n, "mean": mean(d), "wins": sum(x > 0 for x in d)}
    res["sd"] = sd(d)
    res["p_t"] = res["p_sign"] = None
    if n >= 2 and len(set(d)) > 1:
        res["p_t"] = float(stats.ttest_rel([ka[s] for s in seeds], [kb[s] for s in seeds]).pvalue)
    if n >= 1:
        k = sum(x > 0 for x in d)
        nz = sum(x != 0 for x in d)
        res["p_sign"] = float(stats.binomtest(k, nz, 0.5).pvalue) if nz else None
    return res


def support(n: int) -> str:
    if n <= 1:
        return "n=1 (anecdote; no variance estimate)"
    if n == 2:
        return "n=2 (no usable test)"
    if n == 3:
        return "n=3 (direction only; sign test cannot go below p=0.25 two-sided)"
    if n == 4:
        return "n=4 (sign test floor p=0.125 two-sided)"
    return f"n={n} (sign test floor p={2 * 0.5**n:.3f} two-sided)"


# ----------------------------------------------------------------------------------------
# bug exposure
# ----------------------------------------------------------------------------------------
BUG_KEYS = (
    "dac_clip",
    "adc_max",
    "dispatch_precision",
    "remap_bugs12",
    "stuck_weight_coupling",
    "frozen_reference",
    "fault_map_pairing",
)


def _mit_enabled(cfg):
    m = cfg.get("mitigation", {})
    return any(
        m.get(k) for k in ("enable_soft_mitigation", "enable_remapping", "enable_layer_reset")
    )


def exposure(run: Run) -> dict[str, str]:
    cfg = run.cfg
    cb = cfg.get("crossbar", {})
    sim = cfg.get("simulator")
    fault = cfg.get("fault", {})
    has_faults = (fault.get("density") or 0) > 0
    regime = cfg.get("regime")
    mit = cfg.get("mitigation", {})
    t = run.start or run.end
    undated = t is None  # no run.log: pre-provenance commit 85e2c59 (2026-09-28)
    if undated:
        t = T_PREPROV
    e: dict[str, str] = {}
    if sim == "crosssim":
        e["dac_clip"] = "fixed(flag)" if cb.get("calibrate_dac") else "ACTIVE"
        e["adc_max"] = "fixed(flag)" if cb.get("calibrate_adc") else "ACTIVE"
    else:
        e["dac_clip"] = e["adc_max"] = "n/a(aihwkit)"
    soft = measure(run, "soft") or 0
    resets = measure(run, "resets") or 0
    if regime in (None, "continuous") and _mit_enabled(cfg) and has_faults:
        counter = (
            soft > 0 and fault.get("failure_model") == "density" and not cfg.get("secondary_fault")
        )
        if counter:
            e["dispatch_precision"] = "ACTIVE(counter: soft touched cells under density faults)"
        elif undated and (soft or resets or mit.get("enable_remapping")):
            e["dispatch_precision"] = "ACTIVE(pre-provenance, probable)"
        else:
            e["dispatch_precision"] = "fixed"

    else:
        e["dispatch_precision"] = "n/a"
    remap_on = mit.get("enable_remapping") or (
        cfg.get("retraining", {}).get("policy") not in (None, "none")
    )
    if remap_on and has_faults:
        e["remap_bugs12"] = "ACTIVE" if (t and t < T_REMAP_FIX) else "fixed"
    else:
        e["remap_bugs12"] = "n/a"
    if has_faults:
        if regime == "deployment" or cfg.get("retraining", {}).get("freeze_stuck_weights"):
            e["stuck_weight_coupling"] = "fixed(freeze on)"
        else:
            e["stuck_weight_coupling"] = "ACTIVE(freeze off, small)"
    else:
        e["stuck_weight_coupling"] = "n/a"
    if remap_on:
        e["frozen_reference"] = cfg.get("retraining", {}).get("reference_mode", "frozen")
    else:
        e["frozen_reference"] = "n/a"
    if run.events and run.events.get("regime") == "deployment":
        e["fault_map_pairing"] = "hash-verifiable"
    elif has_faults and cfg.get("retraining", {}).get("policy") not in (None, "none"):
        e["fault_map_pairing"] = "seeded-generator(seed*1000+k; policy runner)"
    elif has_faults:
        e["fault_map_pairing"] = "UNVERIFIED(global RNG, system.inject_faults())"
    else:
        e["fault_map_pairing"] = "n/a"
    return e


def exposure_summary(runs: list[Run]):
    act, fixed = set(), set()
    for r in runs:
        for k, v in exposure(r).items():
            if v.startswith(("ACTIVE", "UNVERIFIED")):
                act.add(f"{k}:{v}")
            elif v.startswith(("fixed", "hash")):
                fixed.add(k)
    return "; ".join(sorted(act)), "; ".join(sorted(fixed))


def window(runs: list[Run]) -> str:
    ts = [t for r in runs for t in (r.start, r.end) if t]
    if not ts:
        return "unknown"
    return f"{min(ts):%Y-%m-%d %H:%M}..{max(ts):%Y-%m-%d %H:%M} UTC"


# ----------------------------------------------------------------------------------------
# claims
# ----------------------------------------------------------------------------------------
CLAIMS: list[dict] = []


def tol_status(derived, doc, tol):
    if doc is None:
        return "no-doc-value"
    if derived is None:
        return "NO-SOURCE"
    d = abs(derived - doc)
    if d <= tol:
        return "MATCH"
    if d <= 2 * tol:
        return "ROUNDING"
    return "MISMATCH"


def claim(
    cid,
    area,
    text,
    src,
    runs=None,
    key="final",
    stat="mean",
    doc=None,
    tol=0.05,
    vs=None,
    key_vs=None,
    verdict="VALID",
    superseded_by="",
    note="",
    fixed_derived=None,
    ddof=1,
):
    runs = runs or []
    row = {
        "claim_id": cid,
        "area": area,
        "claim": text,
        "source": src,
        "measuring_point": key if not vs else f"{key} minus {key_vs or key}",
        "doc_value": "" if doc is None else doc,
        "backing_runs": ";".join(r.rel for r in runs),
        "seeds": " ".join(str(r.seed) for r in runs),
        "n": len(runs),
    }
    pr = None
    if vs:
        pr = paired(runs, vs, key, key_vs)
        row["paired_vs"] = ";".join(sorted({r.name for r in vs}))
        row["n"] = pr["n"]
        row["seeds"] = " ".join(str(s) for s in pr["seeds"])
        row["per_seed_diffs"] = " ".join(f"{x:+.2f}" for x in pr["diffs"])
        row["mean_diff"] = None if pr["mean"] is None else round(pr["mean"], 3)
        row["wins"] = f"{pr['wins']}/{pr['n']}"
        row["p_t"] = "" if pr["p_t"] is None else f"{pr['p_t']:.4f}"
        row["p_sign_2s"] = "" if pr["p_sign"] is None else f"{pr['p_sign']:.4f}"
        derived = pr["mean"] if stat in ("mean", "diff") else None
        if stat.startswith("seed:"):
            s = int(stat.split(":")[1])
            derived = dict(zip(pr["seeds"], pr["diffs"])).get(s)
        row["mean"] = row["mean_diff"]
    else:
        v = vals(runs, key)
        row["per_seed"] = " ".join(f"{x:.2f}" for x in v)
        if stat == "mean":
            derived = mean(v)
        elif stat == "sd":
            derived = sd(v, ddof)
        elif stat == "min":
            derived = min(v) if v else None
        elif stat == "max":
            derived = max(v) if v else None
        elif stat == "range":
            derived = (max(v) - min(v)) if v else None
        elif stat.startswith("seed:"):
            s = int(stat.split(":")[1])
            derived = next((measure(r, key) for r in runs if r.seed == s), None)
        else:
            raise ValueError(stat)
        row["mean"] = None if mean(v) is None else round(mean(v), 3)
        row["sd"] = None if sd(v) is None else round(sd(v), 3)
    if fixed_derived is not None:
        derived = fixed_derived
    row["derived_value"] = "" if derived is None else round(derived, 3)
    row["status_vs_doc"] = tol_status(derived, doc, tol)
    allruns = runs + (vs or [])
    act, fixed = exposure_summary(allruns)
    row["bugs_active"] = act
    row["bugs_fixed"] = fixed
    row["run_window"] = window(allruns)
    row["verdict"] = verdict
    row["superseded_by"] = superseded_by
    row["seeds_support"] = support(row["n"])
    row["note"] = note
    CLAIMS.append(row)
    return row


def nosrc(cid, area, text, src, doc, note, verdict="NO-SOURCE"):
    CLAIMS.append(
        {
            "claim_id": cid,
            "area": area,
            "claim": text,
            "source": src,
            "doc_value": doc,
            "status_vs_doc": "NO-SOURCE",
            "verdict": verdict,
            "note": note,
        }
    )


def build_claims() -> None:
    # ---- A. fault-free baselines -------------------------------------------------------
    b4 = RR("resnet9_ibm_recipe_hwa_full_calibrated_gpu")
    claim(
        "A01",
        "baseline",
        "ResNet-9 fault-free final accuracy, 4 seeds",
        "plan 29; MANIFEST row 1",
        b4,
        "final",
        doc=89.79,
        verdict="VALID-WITH-CAVEAT",
        note="last eval (batch 19400) on a 5,000-image test subset (binomial SE 0.43 pt per run); "
        "calibrate_dac/adc on; no device noise, drift or programming error modelled; runs "
        "executed on an uncommitted tree (config carries calibrate_dac, commit 85e2c59 came 3 days "
        "later); no run.log/run_metadata (pre-dates provenance commit)",
    )
    claim(
        "A02",
        "baseline",
        "SD of final accuracy: population (plan 29) vs sample (summarize_results)",
        "plan 29; plan 31",
        b4,
        "final",
        stat="sd",
        doc=0.107,
        tol=0.001,
        ddof=0,
        verdict="VALID",
        note="pop SD 0.107, sample SD 0.124 (SUMMARY.md): both correct, convention "
        "differs; MANIFEST row 1 already states this",
    )
    claim(
        "A03",
        "baseline",
        "SD of final accuracy, sample convention",
        "SUMMARY.md",
        b4,
        "final",
        stat="sd",
        doc=0.124,
        tol=0.001,
        verdict="VALID",
    )
    claim(
        "A04",
        "baseline",
        "max accuracy mean over 4 seeds (whole run)",
        "plan 29; MANIFEST",
        b4,
        "max",
        doc=89.96,
        verdict="VALID-WITH-CAVEAT",
        note="max over 156 analog test evals = best of many looks at the same test subset, "
        "optimistic by construction; the float-pretrain evals (<=88.9) never exceed it here",
    )
    claim(
        "A05",
        "baseline",
        "max accuracy, analog phase only",
        "plan 29",
        b4,
        "max_analog",
        doc=89.96,
        verdict="VALID-WITH-CAVEAT",
        note="identical to A04 for these runs",
    )
    claim(
        "A06",
        "baseline",
        "mean accuracy of the last 5 evals (smoother end point)",
        "derived",
        b4,
        "last5",
        verdict="VALID",
        note="no doc value; reported as the lower-variance alternative",
    )
    for s, dv in ((202609241730, 89.86), (101, 89.62), (102, 89.78), (103, 89.90)):
        claim(
            f"A07.{s}",
            "baseline",
            f"per-seed final accuracy, seed {s}",
            "plan 28/29 table",
            b4,
            "final",
            stat=f"seed:{s}",
            doc=dv,
            verdict="VALID",
        )
    claim(
        "A08",
        "baseline",
        "no accuracy crash at the pretrain->analog transition, all 4 seeds "
        "(first analog eval minus last float eval, worst seed)",
        "plan 29",
        b4,
        "first_analog",
        vs=b4,
        key_vs="last_pretrain",
        verdict="VALID",
        note="negative = a drop at the transition; all 4 drops are < 1 pt (worst -0.74); every "
        "earlier recipe dropped 30-70 pts here",
    )
    nosrc(
        "A09",
        "baseline",
        "Fault-free base net used in the deployment regime is 88.9-89.04 (14 epochs)",
        "plan 38",
        88.84,
        "see number-reconciliation.md (clean chip 89.01 vs 89.79): different "
        "net (10 float + 4 analog epochs) from the 50-epoch baseline; do not mix the two",
        verdict="VALID-WITH-CAVEAT",
    )
    nosrc(
        "A10",
        "baseline",
        "IBM chip 92.81% / digital 93.67% / 1,869,122 params (ResNet-9, Le Gallo 2023)",
        "plan 19, 21; strategy doc 1-2",
        "92.81",
        "external citation, not derivable from results/; "
        "log says verified against IBM blog and arXiv 2212.02872 (not re-fetched here). Our "
        "ResNet-9 has 6,573,130 params / 6,568,640 crossbar weights (3.5x IBM's), 50 epochs not "
        "500, no hardware-aware noise phase, 5,000-image test subset, no device non-idealities: "
        "the 3.0-pt gap to 92.81 is not an apples-to-apples comparison",
        verdict="EXTERNAL",
    )
    # capacity ladder
    claim(
        "A11",
        "capacity",
        "baseline_no_fault (2,000 train / 1,000 test) 43.30 final / 47.30 max",
        "plan 16; SUMMARY",
        R("baseline_no_fault"),
        "final",
        doc=43.30,
        verdict="VALID-WITH-CAVEAT",
        note="CrossSim with DAC clipping + ADC MAX range active; 1,000 test images (SE 1.6 pt)",
    )
    claim(
        "A12",
        "capacity",
        "widened CNN, 16k samples, 30 epochs: 51.70 final / 55.55 max",
        "plan 16",
        R("baseline_no_fault_scaled_check", [42]),
        "final",
        doc=51.70,
        verdict="VALID-WITH-CAVEAT",
        note="numbers real; ladder measured through the buggy "
        "converters, so it does not measure capacity (see A15)",
    )
    claim(
        "A13",
        "capacity",
        "plain 6-conv net regresses vs widened (47.90 / 49.45)",
        "plan 19",
        R("baseline_no_fault_scaled_check_deep", [42]),
        "final",
        doc=47.90,
        verdict="VALID-WITH-CAVEAT",
        note="numbers real; causal reading in plan 19 "
        "('plain-deep optimisation degradation') is INVALIDATED by the later finding that "
        "multiplicative converter error across sequential analog layers hurts deeper stacks "
        "(plan 26, 28); never tested with float or fixed-converter control",
    )
    claim(
        "A14",
        "capacity",
        "ResNet-style 30-epoch fair-schedule result (48.20 / 50.70)",
        "not in the plan log (plan 19 says 'in flight')",
        R("baseline_no_fault_scaled_check_resnet", [42]),
        "final",
        doc=None,
        verdict="VALID-WITH-CAVEAT",
        note="result exists in results/ but is never narrated; same converter caveat",
    )
    claim(
        "A15",
        "capacity",
        "'the bottleneck was model capacity, not training procedure' (plan 16)",
        "plan 16, 'Honest conclusion'",
        R("baseline_no_fault_scaled_check", [42]),
        "final",
        verdict="INVALIDATED",
        superseded_by="plan 23-29 (resnet9_ibm_recipe_hwa_full_calibrated: "
        "89.79 with 3.1M-6.6M weights vs 51.70)",
        note="every ladder row ran through DAC clipping and ADC MAX range; 45.98 -> 89.86 after "
        "the DAC fix with the same architecture. Capacity was never isolated",
    )
    nosrc(
        "A16",
        "capacity",
        "ladder rows 49.50 / 49.75 / 49.25 (data x8, augmentation, wd+cosine)",
        "plan 16 table",
        "49.50/49.75/49.25",
        "intermediate configs overwrote each other; only 43.30 and 51.70 rows have result dirs",
    )
    nosrc(
        "A17",
        "capacity",
        "first GPU run, 6-conv ResNet-style net, full data: 49.5 final / 56.24 max "
        "(baseline_no_fault_full_scale_resnet_gpu_seed42)",
        "plan 20; plan 22; plan 23",
        "49.5/56.24",
        "result dir absent from results/ and from git history (run predates durability fix "
        "plan 31-32); the 3.04M-param net no longer exists in the tree",
    )
    # ResNet-9 failure chain
    claim(
        "A18",
        "failure-chain",
        "SGD one-cycle recipe, analog from scratch: 27.12 final / 30.18 max",
        "plan 22; MANIFEST",
        R("resnet9_ibm_recipe_full_scale_gpu", [42]),
        "final",
        doc=27.12,
        verdict="VALID-WITH-CAVEAT",
        note="grad_clip_norm=1.0 hypothesis (plan 22) was never "
        "tested; the true cause was DAC clipping + ADC compounding (plan 28)",
    )
    claim(
        "A19",
        "failure-chain",
        "same run, max",
        "plan 22",
        R("resnet9_ibm_recipe_full_scale_gpu", [42]),
        "max",
        doc=30.18,
        verdict="VALID",
    )
    claim(
        "A20",
        "failure-chain",
        "Adam isolation run peaks 27.06 at epoch 2, ends 24.14",
        "plan 23",
        R("baseline_no_fault_full_scale_resnet_gpu", [20260922]),
        "max",
        doc=27.06,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="raw max is 28.64 (SUMMARY.md agrees), not 27.06: plan 23 quotes a different point",
    )
    claim(
        "A21",
        "failure-chain",
        "Adam isolation run final 24.14",
        "plan 23; SUMMARY",
        R("baseline_no_fault_full_scale_resnet_gpu", [20260922]),
        "final",
        doc=24.14,
        verdict="VALID",
    )
    claim(
        "A22",
        "failure-chain",
        "hwa-pretrain run 1: float 88.70, analog 31.58 -> 22.70 final",
        "plan 25",
        R("resnet9_ibm_recipe_hwa_pretrain_gpu", [20260923]),
        "final",
        doc=22.70,
        verdict="VALID",
        note="SUMMARY.md max column (88.80 mean) is the FLOAT-phase peak, not an analog number",
    )
    claim(
        "A23",
        "failure-chain",
        "hwa-pretrain run 1 analog-phase max 31.58",
        "plan 25",
        R("resnet9_ibm_recipe_hwa_pretrain_gpu", [20260923]),
        "max_analog",
        doc=31.58,
        verdict="VALID",
    )
    claim(
        "A24",
        "failure-chain",
        "run 2 (finetune lr 0.001): analog peak 33.26, final 25.90",
        "plan 25",
        R("resnet9_ibm_recipe_hwa_pretrain_gpu", [20260923002]),
        "max_analog",
        doc=33.26,
        verdict="VALID-WITH-CAVEAT",
        note="plan 25 quotes the epoch-17 value (batch 7000 = 33.26); the true analog-phase max of "
        "this run is 33.60 (later eval). Immaterial to the conclusion",
    )
    claim(
        "A25",
        "failure-chain",
        "ADC-calibrated combined run: 45.98 final / 46.76 analog max",
        "plan 27; MANIFEST",
        R("resnet9_ibm_recipe_hwa_calibrated_gpu", [202609241000]),
        "final",
        doc=45.98,
        verdict="VALID-WITH-CAVEAT",
        note="SUMMARY.md reports max 88.90 for this run (float-phase peak); the 46.76 quoted in the "
        "log is the analog max (next row); single seed",
    )
    claim(
        "A26",
        "failure-chain",
        "same run, analog-phase max 46.76",
        "plan 27",
        R("resnet9_ibm_recipe_hwa_calibrated_gpu", [202609241000]),
        "max_analog",
        doc=46.76,
        verdict="VALID",
    )
    c1 = R("resnet9_ibm_recipe_hwa_calibrated_gpu", [202609241000])
    claim(
        "A27",
        "failure-chain",
        "DAC fix: 45.98 (calibrated ADC only) -> 89.79 (ADC+DAC, 4 seeds): +43.8",
        "plan 28; plan 29",
        b4,
        "final",
        doc=43.81,
        tol=0.05,
        fixed_derived=mean(vals(b4, "final")) - measure(c1[0], "final"),
        verdict="VALID-WITH-CAVEAT",
        note="UNPAIRED: the 45.98 run is one seed (202609241000) and shares no seed with the 4 fixed "
        "runs; the +43.8 pt effect is ~350x the fixed-arm SD (0.124) so noise is not the question. "
        "Config diff between the two runs is calibrate_dac only (plus seed); the fixed runs executed "
        "on an uncommitted tree three days before the DAC commit 85e2c59",
    )

    claim(
        "A28",
        "dac-probe",
        "local A/B at reduced scale: no DAC fix 24.80 -> with fix 37.40 (seed 7)",
        "plan 28",
        R("local_check_with_dac", [7]),
        "final",
        vs=R("local_check_no_dac", [7]),
        doc=12.6,
        tol=0.05,
        verdict="VALID",
        note="one seed, 500 test images (SE 2.2 pt); the max (42.20) is identical in both because "
        "it is the float-pretrain peak; the fix removes the analog-phase collapse",
    )
    claim(
        "A29",
        "dac-probe",
        "with DAC fix the analog phase is flat (min 36.2 / max 39.0)",
        "plan 28",
        R("local_check_with_dac", [7]),
        "min_analog",
        doc=36.2,
        tol=0.1,
        verdict="VALID",
    )
    claim(
        "A30",
        "dac-probe",
        "without fix the analog phase drifts 16.6-25.8",
        "plan 28",
        R("local_check_no_dac", [7]),
        "min_analog",
        doc=16.6,
        tol=0.1,
        verdict="VALID",
    )
    nosrc(
        "A31",
        "dac-probe",
        "7.57% of every layer's DAC inputs clipped to 1.0",
        "plan 28",
        7.57,
        "instrumentation output was never saved; see --probe-clipping for an independent "
        "order-of-magnitude re-measurement on a trained checkpoint",
        verdict="NO-SOURCE",
    )

    # ---- B. moderate regime (legacy small CNN) -----------------------------------------
    arms = {
        "no_mitigation": ("no_mitigation_moderate", 41.57),
        "soft": ("soft_mitigation_only_moderate", 44.13),
        "remap": ("remapping_only_moderate", 41.90),
        "reset": ("layer_reset_only_moderate", 19.80),
        "combined": ("combined_all_strategies_moderate", 37.93),
    }
    seeds3 = [7, 42, 123]
    g = {k: [r for r in R(v[0]) if r.seed in seeds3] for k, v in arms.items()}
    sup_mod = (
        "resnet9_fault_* (ResNet-9, single seed, current dispatch) and resnet9_fault_mixed_* "
        "(single seed, post remap-fix); no multi-seed rerun on current code exists"
    )
    for k, (nm, dv) in arms.items():
        claim(
            f"B01.{k}",
            "moderate",
            f"{k}: mean final accuracy, 3 seeds (7/42/123)",
            "plan 14; MANIFEST",
            g[k],
            "final",
            doc=dv,
            tol=0.05,
            verdict="SUPERSEDED" if k != "no_mitigation" else "VALID-WITH-CAVEAT",
            superseded_by=sup_mod if k != "no_mitigation" else "",
            note=(
                "old dispatch (acted on heuristic-critical non-stuck cells), DAC clip + ADC MAX, "
                "remap bugs 1-2, epochs differ per arm (15/20/20/20/10), 1,000 test images"
            ),
        )
    claim(
        "B02.no_mitigation.range",
        "moderate",
        "no_mitigation range 39.60-43.40",
        "plan 14",
        g["no_mitigation"],
        "final",
        stat="min",
        doc=39.60,
        verdict="VALID-WITH-CAVEAT",
    )
    claim(
        "B02.reset.range_lo",
        "moderate",
        "layer_reset lowest seed 16.40",
        "plan 14",
        g["reset"],
        "final",
        stat="min",
        doc=16.40,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
    )
    claim(
        "B02.reset.range_hi",
        "moderate",
        "layer_reset highest seed 24.00",
        "plan 14",
        g["reset"],
        "final",
        stat="max",
        doc=24.00,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
    )
    claim(
        "B03.soft_minus_nomit",
        "moderate",
        "soft_mitigation beats no_mitigation by +2.56 pt (3 seeds)",
        "plan 14; MANIFEST 'DEBUNKED explanation'",
        g["soft"],
        "final",
        vs=g["no_mitigation"],
        doc=2.56,
        tol=0.05,
        verdict="INVALIDATED",
        superseded_by="n/a (confounded; never rerun controlled)",
        note="arms differ in epochs (soft 20 vs no_mit 15) and so in total training; soft counter is "
        "25.9-26.1M cell-touches in these runs, so soft DID write to the crossbar (old dispatch "
        "targeted non-stuck critical cells). MANIFEST/plan 30 claim of a 'proven bit-identical "
        "no-op for every config in project history' is false for these runs; it is true only for "
        "the current dispatch (drift_mask) and was verified on ResNet-9 only",
    )
    claim(
        "B04.remap_minus_nomit",
        "moderate",
        "remapping is a wash (+0.33 pt)",
        "plan 14",
        g["remap"],
        "final",
        vs=g["no_mitigation"],
        doc=0.33,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="epochs 20 vs 15; remap pointing at heuristic-critical cells (old dispatch)",
    )
    claim(
        "B05.reset_minus_nomit",
        "moderate",
        "layer_reset trails no_mitigation (-21.8 pt), ranges do not overlap",
        "plan 14; MANIFEST",
        g["reset"],
        "final",
        vs=g["no_mitigation"],
        doc=-21.77,
        tol=0.05,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
        note="reset touched 47-55% of fc1 (heuristic-critical set) in these runs; current dispatch "
        "resets stuck cells only, and on ResNet-9 layer_reset is level with no_mitigation "
        "(B20)",
    )
    claim(
        "B06.combined_minus_nomit",
        "moderate",
        "combined (37.93) below no_mitigation (41.57)",
        "plan 14",
        g["combined"],
        "final",
        vs=g["no_mitigation"],
        doc=-3.64,
        tol=0.05,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
        note="combined runs 10 epochs vs 15",
    )
    claim(
        "B07.health",
        "moderate",
        "final health: no_mitigation 67.07, soft 65.72, remap 65.71, combined 68.51",
        "plan 14",
        g["no_mitigation"],
        "health",
        doc=67.07,
        verdict="VALID",
        note="health is a synthetic stress-model value (plan 36 bug 2), not a physical measurement",
    )
    claim(
        "B08",
        "moderate",
        "first-run layer_reset (whole-layer) 11.80 / combined 33.07",
        "plan 14",
        [],
        doc=11.80,
        verdict="NO-SOURCE",
        note="overwritten by the targeted-reset rerun",
    )
    claim(
        "B09.regime1_no_mit",
        "moderate",
        "regime 1 (density 0.1, interval 8): every arm at chance",
        "plan 13",
        RR("no_mitigation"),
        "final",
        doc=9.30,
        verdict="VALID-WITH-CAVEAT",
        note="regime saturates by construction: 1-0.9^118 = 100% stuck",
    )
    for nm, dv in (
        ("soft_mitigation_only", 8.80),
        ("remapping_only", 7.50),
        ("layer_reset_only", 8.10),
        ("combined_all_strategies", 10.80),
    ):
        claim(
            f"B09.{nm}",
            "moderate",
            f"regime 1 {nm} final {dv}",
            "plan 13",
            R(nm),
            "final",
            doc=dv,
            verdict="VALID-WITH-CAVEAT",
            note="chance level; unseeded dir (seed 42)",
        )
    claim(
        "B10",
        "moderate",
        "fault_aware_retraining 46.40 final (injects once, density 0.1)",
        "plan 13",
        R("fault_aware_retraining"),
        "final",
        doc=46.40,
        verdict="SUPERSEDED",
        superseded_by="co-mitigation deployment-regime arms (plan 35-38); plan 35 itself "
        "calls this config a debunked pinned-cells no-op",
        note="single injection, not comparable",
    )
    claim(
        "B11.manifest_seeds",
        "moderate",
        "MANIFEST: no_mitigation_moderate seeds {42,7,101,102}",
        "MANIFEST row 2",
        g["no_mitigation"],
        "final",
        fixed_derived=3.0,
        doc=4.0,
        tol=0.0,
        verdict="INVALIDATED",
        note="actual seeds are 42/7/123 (+ an unseeded dir that is a copy of seed 42)",
    )
    sm = (REPO / "results" / "SUMMARY.md").read_text()
    for grp, key in (
        ("no_mitigation_moderate", "no_mitigation"),
        ("soft_mitigation_only_moderate", "soft"),
        ("remapping_only_moderate", "remap"),
    ):
        m = re.search(rf"\| `{grp}` \| (\d) \| ([\d.]+)%", sm)
        claim(
            f"B12.summary.{key}",
            "moderate",
            f"SUMMARY.md row {grp}: n={m[1]} mean {m[2]}",
            "results/SUMMARY.md",
            g[key],
            "final",
            doc=float(m[2]),
            tol=0.05,
            verdict="INVALIDATED",
            note="SUMMARY counts the unseeded dir (a duplicate of seed 42) as a 4th seed; true "
            "3-seed mean is the derived value; SUMMARY.md is also stale (generated 2026-09-28)",
        )
    dup = [(r.name, r.n_copies, r.copies_agree) for r in RR("no_mitigation_moderate")]
    CLAIMS[-1]["note"] += f" (dup check: {dup})"
    # devices
    fe = {
        k: RR(f"device_fefet_{k}_moderate")
        for k in ("no_mitigation", "soft_mitigation", "remapping", "layer_reset", "combined")
    }
    claim(
        "C01",
        "device",
        "FeFET no_mitigation 3-seed mean 42.87",
        "plan 15",
        fe["no_mitigation"],
        "final",
        doc=42.87,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
    )
    claim(
        "C02",
        "device",
        "FeFET soft_mitigation 3-seed mean 41.60",
        "plan 15",
        fe["soft_mitigation"],
        "final",
        doc=41.60,
        verdict="SUPERSEDED",
        superseded_by=sup_mod,
    )
    claim(
        "C03",
        "device",
        "FeFET soft minus no_mit (not clearly resolved)",
        "plan 15; MANIFEST",
        fe["soft_mitigation"],
        "final",
        vs=fe["no_mitigation"],
        verdict="VALID-WITH-CAVEAT",
        note="1 of 3 seeds positive; epochs 20 vs 15; the 'open question' is moot because soft "
        "acts on different cells under current code",
    )
    for k, dv in (("remapping", 41.30), ("layer_reset", 13.30), ("combined", 39.50)):
        claim(
            f"C04.{k}",
            "device",
            f"FeFET {k} seed 42 final {dv}",
            "plan 15",
            fe[k],
            "final",
            doc=dv,
            verdict="SUPERSEDED",
            superseded_by=sup_mod,
            note="n=1",
        )
    cs = {
        k: R(f"device_sttmram_{k}_moderate", [42])
        for k in ("no_mitigation", "soft_mitigation", "remapping", "layer_reset", "combined")
    }
    core = {
        "no_mitigation": "no_mitigation_moderate",
        "soft_mitigation": "soft_mitigation_only_moderate",
        "remapping": "remapping_only_moderate",
        "layer_reset": "layer_reset_only_moderate",
        "combined": "combined_all_strategies_moderate",
    }
    same = {k: cs[k][0].acc() == R(core[k], [42])[0].acc() for k in cs}
    claim(
        "C05",
        "device",
        "STT-MRAM on CrossSim is the default device (5 arms bit-identical to core track seed 42)",
        "plan 15; MANIFEST",
        list(itertools.chain.from_iterable(cs.values())),
        "final",
        fixed_derived=float(sum(same.values())),
        doc=5.0,
        tol=0.0,
        verdict="VALID",
        note=f"full accuracy trajectories compared per arm: {same}. Not STT-MRAM evidence",
    )
    ai = R("no_mitigation_aihwkit_moderate", [42])
    claim(
        "C06",
        "backend",
        "AIHWKit no_mitigation seed 42 = 54.60",
        "plan 15; MANIFEST",
        ai,
        "final",
        doc=54.60,
        verdict="VALID-WITH-CAVEAT",
        note="n=1. vs CrossSim 41.70 on the same seed: a 12.9 pt gap that cannot be attributed to the "
        "backend because the CrossSim side carried the DAC-clip and ADC-MAX defects (A15, A27) "
        "which AIHWKit does not have; AIHWKit runs the pure-torch tile (compiled binding broken, "
        "plan 9)",
    )
    claim(
        "C07",
        "backend",
        "AIHWKit minus CrossSim, same seed",
        "derived",
        ai,
        "final",
        vs=R("no_mitigation_moderate", [42]),
        verdict="VALID-WITH-CAVEAT",
        note="not a backend-fidelity measurement",
    )
    mn = R("no_mitigation_mnist_moderate", [42])
    claim(
        "C08",
        "backend",
        "MNIST no_mitigation 86.70 final / 91.20 max, WARN on a 72-pt jump",
        "plan 15",
        mn,
        "final",
        doc=86.70,
        verdict="VALID",
    )
    claim(
        "C09",
        "backend",
        "MNIST largest one-step accuracy jump 72.2",
        "plan 15",
        mn,
        "max_jump",
        doc=72.2,
        tol=0.05,
        verdict="VALID",
    )
    nosrc(
        "C10",
        "backend",
        "full 5-arm comparison on AIHWKit / MNIST",
        "plan 15, 18",
        "outstanding",
        "still outstanding: only no_mitigation exists for both",
        verdict="OPEN",
    )

    # ---- D. ResNet-9 fault campaign, single seed (pre remap fix) ---------------------
    f1 = {
        k: R(f"resnet9_fault_{k}", [42])
        for k in (
            "no_mitigation",
            "soft_mitigation",
            "remapping",
            "layer_reset",
            "combined",
            "soft_mitigation_drift",
        )
    }
    for k, r in f1.items():
        claim(
            f"D01.{k}",
            "resnet9-fault",
            f"resnet9_fault_{k} seed 42: final accuracy",
            "plan 34 (no number logged); derived",
            r,
            "final",
            verdict="SUPERSEDED" if k in ("remapping", "combined") else "VALID-WITH-CAVEAT",
            superseded_by="resnet9_fault_mixed_* / cont_next_classic_* (remap bugs fixed)"
            if k in ("remapping", "combined")
            else "",
            note="n=1; last eval at batch 19400 (3,800 batches after wave 3); max column (~89) is "
            "the pre-fault peak and must not be quoted",
        )
    claim(
        "D02",
        "resnet9-fault",
        "soft_mitigation under density faults is a no-op (soft counter 0, "
        "same code path as no_mitigation)",
        "plan 30",
        f1["soft_mitigation"],
        "soft",
        doc=0.0,
        tol=0.0,
        verdict="VALID",
        note="confirms the no-op for the CURRENT dispatch on "
        "ResNet-9 only (not for the Sep-13 moderate runs, see B03)",
    )
    claim(
        "D03",
        "resnet9-fault",
        "free replicate: soft (no-op) minus no_mitigation, same seed",
        "derived",
        f1["soft_mitigation"],
        "final",
        vs=f1["no_mitigation"],
        verdict="VALID",
        note="the two arms execute the same computation; the difference is run-to-run noise at "
        "n=1 on a 5,000-image test subset (SE 0.7 pt/run)",
    )
    claim(
        "D04",
        "resnet9-fault",
        "remapping fires in one early burst only: 656,868 of 985,293 spares",
        "plan 30 / plan 36 bug 2",
        f1["remapping"],
        "remap",
        doc=656868,
        tol=2.0,
        verdict="VALID",
        note="= 66.7% of capacity: wave 2-3 got none (selection stall + health gate)",
    )
    claim(
        "D05",
        "resnet9-fault",
        "soft_mitigation_drift works against drift (89.32 final)",
        "plan 30 (6th arm); results not narrated",
        f1["soft_mitigation_drift"],
        "final",
        verdict="VALID-WITH-CAVEAT",
        note="no drift-only no_mitigation control exists, so "
        "'works' is unsupported; mixed-run evidence (D14) says drift at this magnitude is benign",
    )
    mx = {
        k: R(f"resnet9_fault_mixed_{k}", [42])
        for k in (
            "no_mitigation",
            "soft_mitigation",
            "layer_reset",
            "remapping",
            "combined",
            "remapping_rowcol",
            "combined_rowcol",
            "remapping_rf05",
            "remapping_rf10",
            "remapping_rf20",
            "remapping_rf30",
        )
    }
    for k, r in mx.items():
        claim(
            f"D10.{k}",
            "mixed",
            f"resnet9_fault_mixed_{k} (density+drift, seed 42): final",
            "SA table 3 / plan 34 items 1-3",
            r,
            "final",
            verdict="VALID-WITH-CAVEAT",
            note="n=1; post remap-fix; freeze OFF; remap order 'prefix'; drift secondary fault",
        )
    claim(
        "D11",
        "mixed",
        "mixed head-to-head: soft/layer_reset/combined within 0.5 pt of no_mitigation",
        "plan 34 item 1",
        mx["soft_mitigation"],
        "final",
        vs=mx["no_mitigation"],
        verdict="VALID-WITH-CAVEAT",
        note="soft minus no_mit; n=1; layer_reset and combined likewise (rows D10)",
    )
    claim(
        "D12",
        "mixed",
        "per-cell remapping (15%) below no_mitigation",
        "derived",
        mx["remapping"],
        "final",
        vs=mx["no_mitigation"],
        verdict="VALID-WITH-CAVEAT",
        note="-2.7 pt at n=1 (SE of the difference ~1 pt); do not over-read",
    )
    claim(
        "D13",
        "mixed",
        "row/col redundancy survives the realistic repair model: +3.1 over per-cell",
        "plan 34 item 2",
        mx["remapping_rowcol"],
        "final",
        vs=mx["remapping"],
        verdict="VALID-WITH-CAVEAT",
        note="n=1",
    )
    claim(
        "D14",
        "mixed",
        "drift as a secondary fault costs ~0.6 pt: mixed no_mit vs continuous classic no_mit",
        "derived",
        mx["no_mitigation"],
        "final",
        vs=R("resnet9_cont_next_classic_no_mitigation", [42]),
        verdict="VALID-WITH-CAVEAT",
        note="-0.58 at n=1 each, SMALLER than the free-replicate spread "
        "(D03: two arms executing the same computation differ by 1.0): drift at this magnitude is "
        "NOT distinguishable from zero. Runs also differ in date (Oct 6-8 vs Oct 9) and in the CUDA "
        "generator fix (42ed9a5)",
    )
    claim(
        "D15",
        "mixed",
        "redundancy sweep: rf05 39.82, rf10 39.98, rf15 39.90, rf20 43.00, rf30 89.14 (SA table)",
        "SA 3 lines 138-141",
        mx["remapping_rf20"],
        "final",
        doc=43.00,
        verdict="VALID-WITH-CAVEAT",
        note="a cliff, not a curve: the spare budget must cover the 27.1% "
        "cumulative stuck fraction (1-0.9^3); nothing between 20% and 30% was run; n=1",
    )
    claim(
        "D16",
        "mixed",
        "rf30 remaps everything and recovers to baseline",
        "SA",
        mx["remapping_rf30"],
        "final",
        doc=89.14,
        verdict="VALID-WITH-CAVEAT",
        note="n=1",
    )

    # ---- E. continuous regime classic arms ---------------------------------------------
    c0 = R("resnet9_cont_next_classic_no_mitigation", [42])
    c15 = R("resnet9_cont_next_classic_remapping", [42])
    c30 = R("resnet9_cont_next_classic_remapping_rf30", [42])
    claim(
        "E01",
        "continuous",
        "classic no mitigation 43.2 (batch 19400)",
        "RR Finding 7",
        c0,
        "final",
        doc=43.2,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="n=1",
    )
    claim(
        "E02",
        "continuous",
        "classic remap 15% spares 44.4",
        "RR Finding 7",
        c15,
        "final",
        doc=44.4,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="n=1; frozen reference + random order",
    )
    claim(
        "E03",
        "continuous",
        "classic remap 30% spares 89.4",
        "RR Finding 7",
        c30,
        "final",
        doc=89.4,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="n=1; 30% >= 27.1% cumulative stuck, so every stuck cell has a spare",
    )
    claim(
        "E04",
        "continuous",
        "classic remap 15% minus no_mitigation",
        "derived",
        c15,
        "final",
        vs=c0,
        verdict="VALID-WITH-CAVEAT",
        note="+1.1 at n=1, inside noise (D03 free replicate = 1.0)",
    )
    dseed = RR("resnet9_cont_d_fixgate", "resnet9_cont_next_d")
    cm = R("resnet9_cont_next_c_living_matched")
    cp = RR("resnet9_cont_c_living", "resnet9_cont_next_c_living")
    claim(
        "E05",
        "continuous",
        "D (gate fixed) end of run 88.7, 3 seeds",
        "RR Finding 7",
        dseed,
        "eor",
        doc=88.7,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="seed 42 end-of-run is the last logged point (batch 19100), 101/102 are summary "
        "end_of_run values; arms do not share a checkpoint (each trains its own net)",
    )
    claim(
        "E06",
        "continuous",
        "C' living (unmatched) end of run 85.2, 3 seeds",
        "RR Finding 7",
        cp,
        "eor",
        doc=85.2,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
    )
    claim(
        "E07",
        "continuous",
        "C'-matched end of run 87.4, 3 seeds",
        "RR Finding 7",
        cm,
        "eor",
        doc=87.4,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
    )
    claim(
        "E08",
        "continuous",
        "D minus classic no_mitigation, seed 42 only",
        "derived",
        [r for r in dseed if r.seed == 42],
        "eor",
        vs=c0,
        key_vs="eor",
        verdict="VALID-WITH-CAVEAT",
        note="paired n=1. CONFOUNDED: C'/D add retraining bursts (burst_batches 390 per wave x 3 waves, "
        "burst lr 0.3-0.5 per config) on top of the same schedule, classic arms get none; ALSO the "
        "fault maps are NOT paired: classic arms inject via system.inject_faults() on the global "
        "RNG, C'/D via the policy runner's seeded generator (seed*1000+k, experiments/run.py:451); no continuous-regime B-matched (periodic burst, no "
        "remap) arm exists, so the gap mixes burst compute/LR with spare allocation",
    )
    claim(
        "E09",
        "continuous",
        "'retraining while faults arrive is what matters' (D and C' far above "
        "classic remapping at 15% spares)",
        "RR Finding 7",
        cm,
        "eor",
        vs=c15,
        key_vs="eor",
        verdict="VALID-WITH-CAVEAT",
        note="paired by seed only (different fault-map RNG, see E08). Numbers right; wording wrong: classic arms also train "
        "continuously through the waves (no_mitigation recovers 10 -> 43). The difference is burst "
        "retraining at lr 0.3-0.5 vs the low one-cycle tail, plus living vs frozen spares; the "
        "burst-matched classical control is missing",
    )
    claim(
        "E10",
        "continuous",
        "3 classic configs exist but were never run: soft_mitigation, "
        "layer_reset, combined (cont_next_classic_*)",
        "experiments/configs",
        [],
        verdict="OPEN",
        note="classic continuous comparison covers 3 of 6 planned arms, all seed 42",
    )
    dnext = RR("resnet9_cont_d_fixgate", "resnet9_cont_next_d")
    claim(
        "E11",
        "continuous",
        "classic remap 30% spares equals D (89.4 vs 89.1), seed 42",
        "RR Finding 7",
        c30,
        "eor",
        vs=[r for r in dnext if r.seed == 42],
        key_vs="eor",
        verdict="VALID-WITH-CAVEAT",
        note="paired n=1 by seed only (classic uses the global RNG for fault maps, D the seeded wave generator, so maps differ); classic rf30 minus D(fixgate). The two arms are NOT comparable in cost: "
        "30% spares >= the 27.1% cumulative stuck fraction (1-0.9^3), i.e. a spare for every stuck "
        "cell by construction; this is a capacity cliff, not a competitive classical mitigation",
    )
    claim(
        "E12",
        "continuous",
        "D beats C'-matched by ~+1.4 pt, 3 of 3 seeds (continuous regime)",
        "RR Finding 7",
        dnext,
        "eor",
        vs=cm,
        key_vs="eor",
        doc=1.4,
        tol=0.05,
        verdict="VALID-WITH-CAVEAT",
        note="D and C'-matched share the seeded wave generator (same maps per seed); the report rounds +1.29 (its own table gives 88.73-87.43 = 1.30) up to +1.4; "
        "n=3, 3/3 positive but two-sided sign p floor 0.25; paired t "
        "reported; seed 42 D is the earlier 'fixgate' config, seeds 101/102 are 'next_d' (same policy, "
        "different launch)",
    )
    claim(
        "E13",
        "continuous",
        "classic no_mitigation recovers from chance after every wave "
        "(training continues through faults)",
        "plan 36",
        c0,
        "final",
        verdict="VALID",
        note="trajectory seed 42 (sampled every 800 batches): 88.7 -> 32 (400 batches after wave 1) -> 65.6 -> 10.2 (wave 2) -> 59.4 -> 30.4 (wave 3) "
        "-> 43.4; each wave is a fresh collapse, plateaus fall with cumulative stuck fraction",
    )
    # ---- F. deployment regime classical-like arms ---------------------------------------
    r0 = R("resnet9_deploy_r0")
    r1 = RR("resnet9_deploy_r1", "resnet9_final_r1")
    claim(
        "F01",
        "deployment",
        "R0 (no mitigation) ends near chance: 10.7 mean, 3 seeds",
        "plan 38",
        r0,
        "ev_final",
        doc=10.7,
        tol=0.05,
        verdict="VALID",
        note="inference-only lifetime, "
        "no retraining; fault maps hash-verified identical across arms",
    )
    claim(
        "F02",
        "deployment",
        "R1 (remap only, 15% spares) 10.3 mean (3 seeds, plan 38)",
        "plan 38",
        R("resnet9_deploy_r1"),
        "ev_final",
        doc=10.3,
        tol=0.05,
        verdict="VALID",
    )
    claim(
        "F03",
        "deployment",
        "R1 5-seed mean (3 + seeds 103/104)",
        "stats-final RT.D-R1.main",
        r1,
        "ev_final",
        doc=9.732,
        tol=0.001,
        verdict="VALID",
        note="seeds 103/104 ran on a "
        "different box/GPU (fault maps differ from 42/101/102, equal within each seed's arms)",
    )
    claim(
        "F04",
        "deployment",
        "R1 minus R0 (remap-only adds nothing at 15% spares)",
        "derived",
        R("resnet9_deploy_r1"),
        "ev_final",
        vs=r0,
        verdict="VALID",
        note="3 seeds, direction negative and tiny",
    )
    claim(
        "F05",
        "deployment",
        "R1 restores wave-0 faults exactly (88.9) then collapses when spares run out",
        "derived from events.json",
        R("resnet9_deploy_r1", [42]),
        "ev_final",
        verdict="VALID",
        note="see wave trajectory table printed by this script",
    )
    for nm in ("rf02", "rf05", "rf10", "rf15", "rf30"):
        rr_ = R(f"resnet9_deploy_r1_stress_{nm}")
        claim(
            f"F06.{nm}",
            "deployment",
            f"R1 spare sweep {nm}: final accuracy",
            "plan 38 / RR Finding 8",
            rr_,
            "ev_final",
            verdict="VALID",
            note="3 seeds; rf30 equals the clean chip",
        )
    bo = RR("resnet9_deploy_b")
    claim(
        "F07",
        "deployment",
        "B (periodic burst) at the pre-tuning rate: 38.6",
        "plan 38",
        bo,
        "ev_final",
        doc=38.6,
        tol=0.05,
        verdict="SUPERSEDED",
        superseded_by="resnet9_deploy_b_eta5e-1 (tuned rate)",
        note="burst lr 0.005 sat at the edge of the calibration grid; B is +44.7 pt at the tuned rate",
    )
    claim(
        "F08",
        "deployment",
        "B tuned minus B old rate (3 seeds)",
        "stats-final LR.ResNet-9.B",
        R("resnet9_deploy_b_eta5e-1"),
        "ev_final",
        vs=bo,
        verdict="VALID",
        note="the 'D 83.3 vs C' 49.3, B 38.6' headline (plan 38, 39) was a tuning artefact of the "
        "classical baselines",
    )
    cfr, cli = RR("resnet9_deploy_c"), RR("resnet9_deploy_c_living")
    claim(
        "F09",
        "deployment",
        "C' (living) minus C (frozen reference), old rate: +2.5 (49.3 vs 46.8)",
        "plan 38",
        cli,
        "ev_final",
        vs=cfr,
        doc=2.5,
        tol=0.1,
        verdict="VALID-WITH-CAVEAT",
        note="3 seeds; stale-reference fix is the only config difference; not rerun at the tuned rate",
    )
    d5 = RR("resnet9_deploy_d_eta3e-1", "resnet9_final_d")
    bm5 = RR("resnet9_deploy_b_matched_eta5e-1", "resnet9_final_b_matched")
    cm5 = RR("resnet9_deploy_c_living_matched_eta5e-1", "resnet9_final_c_living_matched")
    claim(
        "F10",
        "deployment",
        "B-matched (tuned) 5-seed mean",
        "stats-final F4",
        bm5,
        "ev_final",
        doc=84.72,
        tol=0.05,
        verdict="VALID",
    )
    claim(
        "F11",
        "deployment",
        "C'-matched (tuned) 5-seed mean",
        "stats-final F4",
        cm5,
        "ev_final",
        doc=87.44,
        tol=0.05,
        verdict="VALID",
    )
    claim(
        "F12",
        "deployment",
        "D 5-seed mean",
        "stats-final F4",
        d5,
        "ev_final",
        doc=88.5,
        tol=0.05,
        verdict="VALID",
    )
    claim(
        "F13",
        "deployment",
        "C'-matched minus B-matched (5 seeds): +2.7",
        "stats-final F4.CM-BM",
        cm5,
        "ev_final",
        vs=bm5,
        doc=2.72,
        tol=0.05,
        verdict="VALID",
    )
    claim(
        "F14",
        "deployment",
        "B-matched minus R1: retraining without remap at equal compute",
        "derived",
        bm5,
        "ev_final",
        vs=r1,
        verdict="VALID",
    )
    claim(
        "F15",
        "deployment",
        "C'-matched minus R1",
        "derived",
        cm5,
        "ev_final",
        vs=r1,
        verdict="VALID",
    )
    claim(
        "F16",
        "deployment",
        "D minus C'-matched (5 seeds) +1.06: the cleanest classical-vs-D gap",
        "stats-final F4.D-CM",
        d5,
        "ev_final",
        vs=cm5,
        doc=1.06,
        tol=0.05,
        verdict="VALID",
        note="5/5 seeds positive; Holm-adjusted p 0.056 in stats-final",
    )
    claim(
        "F17",
        "deployment",
        "D minus R1 (5 seeds) +78.8",
        "stats-final RT.D-R1.main",
        d5,
        "ev_final",
        vs=r1,
        doc=78.77,
        tol=0.05,
        verdict="VALID",
    )


def build_more_claims() -> None:
    """G: classical-like R1 on other architectures / datasets.  H: the bug catalogue as claims."""
    for nm, label in (
        ("vgg6_net2_deploy_r1", "VGG-6"),
        ("vgg6_final_deploy_r1", "VGG-6 (extra seeds 103/104)"),
        ("vit_net3_deploy_r1", "ViT-tiny"),
        ("vit_final_deploy_r1", "ViT-tiny (extra seeds)"),
        ("resnet9_next_deploy_r1_c100", "ResNet-9 CIFAR-100"),
        ("resnet9_deploy_r1_clu", "ResNet-9 clustered faults"),
    ):
        claim(
            f"G.{nm}",
            "deployment",
            f"R1 (remap only, 15% spares) on {label} ends at chance",
            "RR Finding 8 / plan 38",
            R(nm),
            "ev_final",
            verdict="VALID-WITH-CAVEAT" if "clu" in nm else "VALID",
            note=(
                "run dated 2026-10-08 06:57 UTC, 5 h BEFORE the clustered-selection fix "
                "commit 42ed9a5 (bias towards low rows); n=1"
                if "clu" in nm
                else "chance = 10% (1% for CIFAR-100); R1 spends the spare budget in wave 0/1 and collapses"
            ),
        )

    # ---- H. bug catalogue: measured evidence
    seeds3 = [7, 42, 123]
    g = {
        "soft": [r for r in R("soft_mitigation_only_moderate") if r.seed in seeds3],
    }
    mx = {k: R(f"resnet9_fault_mixed_{k}", [42]) for k in ("no_mitigation", "soft_mitigation")}
    bo = RR("resnet9_deploy_b")
    bt = RR("resnet9_deploy_b_eta5e-1", "resnet9_final_b")
    cfr, cli = RR("resnet9_deploy_c"), RR("resnet9_deploy_c_living")
    rm_old, rm_new = R("resnet9_fault_remapping", [42]), R("resnet9_fault_mixed_remapping", [42])
    claim(
        "H01",
        "bug:remap",
        "remap bugs 1+2: pre-fix run spent 656,868 of 985,293 spares (66.7%)",
        "plan 36; MANIFEST 2026-10-02",
        rm_old,
        "remap",
        doc=656868,
        tol=2.0,
        verdict="VALID",
        note="run.log: wave 1 'Remapped 14745/14745 (7373 remaining)', wave 2 'Remapped 0/14745'",
    )
    claim(
        "H02",
        "bug:remap",
        "post-fix runs saturate every pool at exactly 15% (985,293)",
        "plan 36",
        rm_new,
        "remap",
        doc=985293,
        tol=1.0,
        verdict="VALID",
        note="run.log wave 2: 'Remapped 87/172 ... (0 capacity remaining)'; wave 3 gets nothing "
        "because the pool is empty, not because of the gate",
    )
    claim(
        "H03",
        "bug:dispatch",
        "soft_mitigation touched 25.9-26.1M cells in the moderate runs "
        "(density faults) but 0 on ResNet-9 (current dispatch)",
        "derived; plan 14 vs plan 30",
        g["soft"],
        "soft",
        doc=26.0e6,
        tol=0.2e6,
        verdict="VALID",
        note="refutes the 'bit-identical no-op for every config in history' wording of plan 30/MANIFEST: "
        "the pre-fix dispatch applied soft to heuristic-critical cells; drift_mask filter landed "
        "between 2026-09-13 runs and the 0e873e8 commit (2026-09-14)",
    )
    claim(
        "H04",
        "bug:dispatch",
        "mixed-fault soft_mitigation touches 113M cells (drift channel) yet equals "
        "no_mitigation (+0.12)",
        "derived",
        mx["soft_mitigation"],
        "soft",
        doc=113.4e6,
        tol=0.2e6,
        verdict="VALID",
        note="soft DOES fire under the drift channel and still does not move accuracy: "
        "the drift model at density 0.1 is benign for this network (D14)",
    )
    claim(
        "H05",
        "bug:stale-reference",
        "frozen vs living spare reference: +2.5 pt (C' vs C), 3/3 seeds",
        "plan 38",
        cli,
        "ev_final",
        vs=cfr,
        doc=2.5,
        tol=0.1,
        verdict="VALID-WITH-CAVEAT",
        note="measured at the OLD burst rate; not re-measured at the tuned rate",
    )
    claim(
        "H06",
        "bug:tuning",
        "classical B baseline was under-tuned: +44.7 pt from burst lr 0.005 to 0.5",
        "stats-final LR.ResNet-9.B",
        bt,
        "ev_final",
        vs=bo,
        doc=44.7,
        tol=0.1,
        verdict="VALID",
        note="the largest single effect in the whole classical comparison is a baseline-tuning one; "
        "LR chosen on seed 42 test curves (open user decision #1)",
    )
    ab = RR("resnet9_ibm_recipe_hwa_calibrated_gpu")
    claim(
        "H07",
        "bug:dac",
        "no accuracy crash at pretrain->analog handover after the DAC fix; before it "
        "the same handover fell 88.90 -> 18.54",
        "plan 27/28/29",
        ab,
        "min_analog",
        vs=ab,
        key_vs="last_pretrain",
        doc=-70.36,
        tol=0.1,
        verdict="VALID-WITH-CAVEAT",
        note="ADC-only run 202609241000: last float eval 88.90, analog min 16.14 (batch 4000): -72.8. "
        "plan 27 quotes 18.54, which is the 4th post-handover eval (batch 4200), not the minimum. "
        "Fixed-DAC runs: worst handover drop is -0.94 (A08)",
    )
    claim(
        "H08",
        "bug:adc",
        "ADC MAX range leaves ~5.6 effective bits (27 rows) and ~2 (4608 rows) of 8",
        "plan 26 fork 1",
        [],
        doc=None,
        verdict="VALID",
        fixed_derived=8 - math.log2(math.sqrt(4608)),
        note="re-derived: MAX range scales with N, typical |sum| with sqrt(N): bits lost = log2(sqrt(N)) "
        "= 2.38 (N=27) and 6.08 (N=4608), so 5.6 and 1.9 remain. Matches the log",
    )


# ----------------------------------------------------------------------------------------
# extra tables (stdout)
# ----------------------------------------------------------------------------------------
def pairing_ok(a: Run, b: Run) -> str:
    ea, eb = a.events, b.events
    if not ea or not eb:
        return "no-events"
    ha = [e.get("fault_map_hash") for e in ea.get("events", [])]
    hb = [e.get("fault_map_hash") for e in eb.get("events", [])]
    n = min(len(ha), len(hb))
    return "same-maps" if n and ha[:n] == hb[:n] else "DIFFERENT-maps"


def print_tables() -> None:
    print("\n== R1 / R0 / B-matched pairing by fault-map hash (per seed) ==")
    bm_map = {r.seed: r for r in bm5_runs()}
    for r in RR("resnet9_deploy_r1", "resnet9_final_r1"):
        b = bm_map.get(r.seed)
        r0 = next((x for x in R("resnet9_deploy_r0") if x.seed == r.seed), None)
        print(
            f"seed {r.seed}: R1 vs B-matched {pairing_ok(r, b) if b else 'n/a'}; "
            f"R1 vs R0 {pairing_ok(r, r0) if r0 else 'n/a'}"
        )
    print("\n== R1 wave trajectory (test_acc_after per wave) ==")
    for r in RR("resnet9_deploy_r1", "resnet9_final_r1"):
        ev = (r.events or {}).get("events", [])
        print(
            r.seed,
            [e.get("test_acc_after") for e in ev],
            "remapped",
            [e.get("remapped_this_wave") for e in ev],
            "base",
            (r.events or {}).get("baseline_test_acc"),
        )
    print("\n== moderate regime per-seed finals / epochs ==")
    for nm in (
        "no_mitigation_moderate",
        "soft_mitigation_only_moderate",
        "remapping_only_moderate",
        "layer_reset_only_moderate",
        "combined_all_strategies_moderate",
    ):
        for r in R(nm):
            print(
                f"{nm:34s} s{r.seed:<4d} final {measure(r, 'final'):5.1f} epochs {r.cfg['epochs']} "
                f"interval {r.cfg['fault']['injection_interval']} soft {measure(r, 'soft'):.0f} "
                f"remap {measure(r, 'remap'):.0f} resets {measure(r, 'resets'):.0f} copies {r.n_copies} "
                f"agree {r.copies_agree} end {r.end.strftime('%m-%d %H:%M') if r.end else 'n/a'}"
            )
    print("\n== exposure of selected runs ==")
    for nm in (
        "no_mitigation_moderate",
        "resnet9_fault_remapping",
        "resnet9_fault_mixed_remapping",
        "resnet9_cont_next_classic_remapping",
        "resnet9_deploy_r1",
        "resnet9_ibm_recipe_hwa_full_calibrated_gpu",
    ):
        r = next(iter(R(nm)), None)
        if r:
            print(nm, r.start, r.end, r.time_source, exposure(r))
    print("\n== phase of cumulative stuck fraction ==")
    for k in (1, 2, 3, 118):
        print(k, "waves:", f"{1 - 0.9**k:.6f}")
    print(
        "ADC MAX effective bits lost: log2(sqrt(N)) for N=27,4608:",
        round(math.log2(math.sqrt(27)), 2),
        round(math.log2(math.sqrt(4608)), 2),
    )


def bm5_runs():
    return RR("resnet9_deploy_b_matched_eta5e-1", "resnet9_final_b_matched")


# ----------------------------------------------------------------------------------------
# optional CPU probes
# ----------------------------------------------------------------------------------------
def probe_legacy_dac() -> None:
    """A/B calibrate_dac on the original Sep-11 SimpleCNN, fault-free, 3 epochs, seeds 42 and 7."""
    sys.path.insert(0, str(REPO))
    import types

    import experiments.run as runmod
    from neurofault.config import load_config

    src = subprocess.run(
        ["git", "show", "a781a1e:src/neurofault/models/cnn.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mod = types.ModuleType("old_cnn")
    exec(src, mod.__dict__)  # noqa: S102 - our own committed file
    runmod.build_model = lambda cfg: mod.SimpleCNN()
    for seed in (42, 7):
        for dac in (False, True):
            cfg = load_config(str(REPO / "experiments/configs/no_mitigation_moderate.yaml"))
            cfg.fault.density = 0.0
            cfg.epochs = 3
            cfg.seed = seed
            cfg.crossbar.calibrate_dac = dac
            m = runmod.run_experiment(cfg)
            print(
                f"legacy-cnn seed {seed} calibrate_dac={dac}: last evals {m['accuracy'][-4:]} "
                f"grad_norm tail {[round(x) for x in m['grad_norm'][-3:]]}"
            )


def probe_clipping() -> None:
    """Fraction of inputs to every Conv/Linear of a trained ResNet-9 with |x| > 1 (float model)."""
    import torch

    from neurofault.config import ExperimentConfig
    from neurofault.data import load_datasets
    from neurofault.models.cnn import build_model

    ck = REPO / "results/_live2/resnet9_predeploy_seed42/pre_deploy.pt"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = ExperimentConfig(name="clip_probe", test_samples=512, train_samples=512, batch_size=128)
    model = build_model(cfg)
    sd = (
        blob.get("model_state_dict", blob.get("state_dict", blob))
        if isinstance(blob, dict)
        else blob
    )
    model.load_state_dict(sd)
    model.eval()
    _, test = load_datasets(cfg)
    stats_: dict[str, list[float]] = {}

    def hook(name):
        def f(_m, inp, _out):
            x = inp[0].detach()
            stats_.setdefault(name, []).append(float((x.abs() > 1.0).float().mean()))

        return f

    for n, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            m.register_forward_hook(hook(n))
    with torch.no_grad():
        for i, (x, _y) in enumerate(test):
            model(x)
            if i >= 3:
                break
    per = {k: mean(v) for k, v in stats_.items()}
    for k, v in per.items():
        print(f"{k:28s} frac(|x|>1) = {v:.4f}")
    print(
        "mean over layers",
        round(mean(list(per.values())), 4),
        "excluding first layer",
        round(mean([v for k, v in per.items() if k != next(iter(per))]), 4),
    )


def write_csv() -> None:
    cols = [
        "claim_id",
        "area",
        "claim",
        "source",
        "doc_value",
        "derived_value",
        "status_vs_doc",
        "measuring_point",
        "backing_runs",
        "seeds",
        "n",
        "per_seed",
        "mean",
        "sd",
        "paired_vs",
        "per_seed_diffs",
        "mean_diff",
        "wins",
        "p_t",
        "p_sign_2s",
        "run_window",
        "bugs_active",
        "bugs_fixed",
        "verdict",
        "superseded_by",
        "seeds_support",
        "note",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "classical_claims.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in CLAIMS:
            w.writerow(row)


_RAW_NEEDED = (
    "needs the raw run archive (per-run run.log, checkpoint .pt and validation_report.json files), "
    "which is not shipped in this repository. Its outputs are shipped instead under "
    "results/derived/ (classical_claims.csv). Nothing was run and nothing was written."
)


def _raw_archive_present() -> bool:
    """True when per-run run.log files exist under results/ (not shipped in the public repo)."""
    return any(RESULTS.glob("*/run.log")) or any(RESULTS.glob("*/*/run.log"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-legacy-dac", action="store_true")
    ap.add_argument("--probe-clipping", action="store_true")
    args = ap.parse_args()
    if not (args.probe_legacy_dac or args.probe_clipping) and not _raw_archive_present():
        print("audit_classical.py " + _RAW_NEEDED)
        return
    if args.probe_legacy_dac:
        probe_legacy_dac()
        return
    if args.probe_clipping:
        probe_clipping()
        return
    build_claims()
    build_more_claims()
    write_csv()
    for c in CLAIMS:
        print(
            f"{c['claim_id']:24s} {c.get('verdict', ''):20s} {c.get('doc_value', '')!s:>8} "
            f"{c.get('derived_value', '')!s:>8} {c.get('status_vs_doc', ''):10s} "
            f"n={c.get('n', '')} diffs={c.get('per_seed_diffs', '')} p_t={c.get('p_t', '')}"
        )
    print_tables()
    from collections import Counter

    print(Counter(c["verdict"] for c in CLAIMS), len(CLAIMS))
    print(Counter(c["status_vs_doc"] for c in CLAIMS))


if __name__ == "__main__":
    main()

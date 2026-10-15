"""Read-only validity and provenance audit of every run under results/ (not results/spectral).

Never writes into raw run directories. It imports the validation library (neurofault.validation) and
re-runs the checks in memory; outputs go to results/derived/ (requires the raw run archive, see main()):

  run_audit.csv    one row per run directory (validation, provenance, config integrity, fingerprint)
  pair_audit.csv   one row per (comparison, seed): do the two arms share checkpoint + fault maps?
  dup_audit.csv    one row per (run, seed) present in more than one results dir
  group_audit.csv  one row per experiment group (aggregated flags)
  audit_summary.json  counts used in run-audit.md

Usage: uv run python scripts/analysis/audit_runs.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from neurofault.config import ExperimentConfig, load_config
from neurofault.validation.results_validator import Level, overall_verdict, validate

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
CONFIGS = REPO / "experiments" / "configs"
OUT = REPO / "results" / "derived"

IST = timezone(timedelta(hours=5, minutes=30))
# Code-change dates (commit time; the working tree may have held the change earlier, so a run
# dated BEFORE the commit is only "possibly pre-fix"). UTC.
FIXES = {
    "dac_clip_fix_2026-09-28": datetime(2026, 9, 28, 21, 27, tzinfo=IST),
    "remap_bugs_fix_2026-10-02": datetime(2026, 10, 2, 22, 56, tzinfo=IST),
    "deploy_regime_freeze_2026-10-03": datetime(2026, 10, 3, 0, 47, tzinfo=IST),
    "clustered_cuda_gate_fixes_2026-10-08": datetime(2026, 10, 8, 17, 28, tzinfo=IST),
}
# Config fields added by uncommitted working-tree edits (see git diff) - a run whose stored
# config lacks them ran on code without them; a run that has them cannot be reproduced from HEAD.
UNCOMMITTED_FIELDS = (
    "fault.stuck_lrs_fraction",
    "retraining.gate_noise_k",
    "retraining.noise_aware_plateau",
    "retraining.anneal_on_stop",
    "retraining.restore_best",
    "retraining.anneal_steps",
)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = json.dumps(v, sort_keys=True) if isinstance(v, (list, tuple)) else v
    return out


def _parse_ts(line: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return (
        datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if m
        else None
    )


def _log_window(path: Path):
    """First/last timestamp in run.log (box clock, assumed UTC; unverified)."""
    log = path / "run.log"
    if not log.exists():
        return None, None
    first = last = None
    with log.open(errors="replace") as fh:
        for line in fh:
            ts = _parse_ts(line)
            if ts:
                first = first or ts
                last = ts
    return first, last


def discover():
    dirs = []
    for d in sorted(RESULTS.iterdir()):
        if d.is_dir() and d.name.startswith("_live"):
            dirs += [(d.name, c) for c in sorted(d.iterdir()) if c.is_dir()]
        elif d.is_dir() and not d.name.startswith("_") and d.name != "spectral":
            dirs.append(("top", d))
    return [(live, p) for live, p in dirs if (p / "config.json").exists()]


def group_of(base: str) -> str:
    rules = [
        (r"^resnet9_calib_", "S4 calibrated faults (old D vs B/C')"),
        (r"^resnet9_d2calib_", "S4 calibrated faults (D2/C2)"),
        (r"^resnet9_d2_", "D2 main test (5 seeds)"),
        (r"^resnet9_real_", "Realistic fault models S1-S3"),
        (r"^resnet9_final_deploy_c_living_c100", "CIFAR-100 final"),
        (r"^resnet9_final_", "ResNet-9 final main test"),
        (r"^vgg6_final_", "VGG6 final"),
        (r"^vit_final_", "ViT final"),
        (r"^vgg6_", "VGG6 net2"),
        (r"^vit_", "ViT net3"),
        (r"^resnet9_next_", "CIFAR-100 next"),
        (r"^resnet9_cont_", "Continuous regime"),
        (r"^resnet9_deploy_liu_style", "Liu-style baseline"),
        (r"^resnet9_deploy_.*_c100", "CIFAR-100 deploy"),
        (r"^resnet9_deploy_.*_stress", "Stress sweeps"),
        (r"^resnet9_deploy_.*_eta", "Burst-LR sweeps"),
        (r"^resnet9_deploy_.*(_g\d|_gate|gfix|matchDgfix)", "Gate / priors variants"),
        (r"^resnet9_deploy_.*(_rf\d|_clu)", "Other fault patterns (seed 42)"),
        (r"^resnet9_deploy_(s2_|d_ew)", "Held-out schedule / wave-count"),
        (r"^resnet9_deploy_", "Stage 1/2 (deploy_*)"),
        (r"predeploy", "Pre-deploy checkpoints"),
        (r"^resnet9_fault_", "Early continuous fault runs (legacy)"),
        (
            r"^resnet9_ibm|^baseline|^smoke|^local_check|^pretrain|^calibrate|^combined_smoke",
            "Baseline/smoke",
        ),
    ]
    for pat, label in rules:
        if re.search(pat, base):
            return label
    return "Legacy Sept (pre-ResNet9 / moderate)"


def build_yaml_index():
    idx, bad = {}, {}
    for y in sorted(CONFIGS.glob("*.yaml")):
        if "spectral" in y.name:
            continue
        try:
            cfg = load_config(y)
            idx.setdefault(cfg.name, []).append((y, cfg))
        except Exception as exc:  # noqa: BLE001 - report, keep going
            bad[y.name] = repr(exc)[:120]
    return idx, bad


def config_compare(stored: dict, base: str, idx):
    """Compare stored config.json to the current yaml (name/seed/pre_deploy_checkpoint ignored)."""
    if base not in idx:
        return {"yaml": "", "cfg_status": "no_yaml", "cfg_diff": "", "cfg_added_nondefault": ""}
    y, cfg = idx[base][0]
    cur = _flat(asdict(cfg))
    sto = _flat(stored)
    default = _flat(asdict(ExperimentConfig(name="x")))
    sec_default = _flat(asdict(ExperimentConfig(name="x").fault), "secondary_fault.")
    default.update({k: v for k, v in sec_default.items() if k not in default})
    skip = {"name", "seed", "pre_deploy_checkpoint"}
    diff, added_nd = [], []
    for k in sorted(set(cur) | set(sto)):
        if k in skip:
            continue
        if k not in sto:
            if cur[k] != default.get(k):
                added_nd.append(f"{k}={cur[k]}")
        elif k not in cur:
            diff.append(f"{k}: stored={sto[k]} yaml=<removed>")
        elif cur[k] != sto[k]:
            diff.append(f"{k}: stored={sto[k]} yaml={cur[k]}")
    status = "match" if not diff and not added_nd else "differs"
    return {
        "yaml": y.name,
        "cfg_status": status,
        "cfg_diff": " | ".join(diff),
        "cfg_added_nondefault": " | ".join(added_nd),
    }


def fix_flags(cfg: dict, start: datetime | None, kind: str):
    flags = []
    fault = cfg.get("fault", {})
    mit = cfg.get("mitigation", {})
    if start is None:
        return ["no_run_log_timestamp"]
    t = start
    if t < FIXES["dac_clip_fix_2026-09-28"]:
        flags.append("predates_DAC_clip_fix(all accuracy numbers)")
    if mit.get("enable_remapping") and t < FIXES["remap_bugs_fix_2026-10-02"]:
        flags.append("predates_remap_bugs_fix(remap/combined arms invalid)")
    if cfg.get("regime") == "continuous" and t < FIXES["deploy_regime_freeze_2026-10-03"]:
        flags.append("continuous_regime_no_stuck_freeze(spurious coupling caveat)")
    if t < FIXES["clustered_cuda_gate_fixes_2026-10-08"]:
        if fault.get("distribution") == "clustered":
            flags.append("clustered_selection_bias_fix_maybe_missing")
        if (
            cfg.get("regime") == "continuous"
            and cfg.get("retraining", {}).get("policy", "none") != "none"
        ):
            flags.append("continuous_policy_bridge_before_gate_ref_fix")
    return flags


def schema_info(cfg: dict):
    flat = _flat(cfg)
    have = [f for f in UNCOMMITTED_FIELDS if f in flat]
    return ";".join(have)


def needs_uncommitted(cfg: dict) -> str:
    """Features of this run's config that exist only in the uncommitted working-tree edits."""
    why = []
    if cfg.get("model_arch") in ("vgg6", "vit_tiny"):
        why.append("arch=" + cfg["model_arch"])
    for key in ("fault", "secondary_fault"):
        f = cfg.get(key) or {}
        if f.get("distribution") == "line":
            why.append(f"{key}.distribution=line")
        if f.get("stuck_lrs_fraction") not in (None, 0.5):
            why.append(f"{key}.stuck_lrs_fraction={f['stuck_lrs_fraction']}")
    if cfg.get("optimizer") == "adamw":
        why.append("optimizer=adamw")
    rt = cfg.get("retraining") or {}
    if rt.get("policy") == "capacity_or_canary":
        why.append("policy=capacity_or_canary")
    if (
        rt.get("gate_noise_k") is not None
        or rt.get("noise_aware_plateau")
        or rt.get("anneal_on_stop")
        or rt.get("restore_best")
    ):
        why.append("D2_switches")
    return ";".join(why)


def code_generation(cfg: dict) -> str:
    """Lower bound on the code version, read from which config fields the run's code knew about."""
    flat = _flat(cfg)
    if any(f in flat for f in UNCOMMITTED_FIELDS):
        return "gen3_uncommitted_working_tree(stuck_lrs_fraction/D2 fields)"
    if "num_classes" in flat or "retraining.canary_gate_reference" in flat:
        return "gen2_post_42ed9a5(cifar100,gate_ref)"
    if "retraining.policy" in flat:
        return "gen1_deployment_regime"
    return "gen0_pre_deployment_regime"


def fingerprint(events: dict | None):
    if not events:
        return {}
    evs = events.get("events", [])
    return {
        "ckpt_hash": (events.get("checkpoint_config_hash") or "")[:12],
        "ckpt_acc": events.get("checkpoint_final_test_acc"),
        "base_test": events.get("baseline_test_acc"),
        "base_canary": events.get("baseline_canary_acc"),
        "n_events": len(evs),
        "wave_hashes": json.dumps(
            [
                hashlib.md5(json.dumps(e["fault_map_hash"], sort_keys=True).encode()).hexdigest()[
                    :8
                ]
                for e in evs
            ]
        ),
        "densities": json.dumps([e.get("density") for e in evs]),
    }


def audit_rows():
    idx, bad_yaml = build_yaml_index()
    rows = []
    for live, p in discover():
        cfg = _load(p / "config.json")
        metrics = _load(p / "metrics.json")
        events = _load(p / "events.json")
        stored_rep = _load(p / "validation_report.json")
        meta = _load(p / "run_metadata.json") or {}
        m = re.match(r"(.*)_seed(\d+)$", p.name)
        base, seed = (m.group(1), int(m.group(2))) if m else (p.name, cfg.get("seed"))
        regime = cfg.get("regime", "continuous")
        if events is not None:
            kind = "deployment"
        elif regime == "deployment" and cfg.get("save_pre_deploy_checkpoint"):
            kind = "predeploy"
        else:
            kind = regime
        row = {
            "path": str(p.relative_to(REPO)),
            "live": live,
            "run": base,
            "seed": seed,
            "group": group_of(base),
            "kind": kind,
            "arch": cfg.get("model_arch"),
            "dataset": cfg.get("dataset"),
            "complete": (events or {}).get("complete")
            if events is not None
            else (metrics is not None),
            "has_metrics": metrics is not None,
        }
        # 1. validation re-check
        if metrics is not None:
            try:
                res = validate(metrics, cfg, results_name=p.name, events=events)
                row["recheck"] = overall_verdict(res).value
                nonpass = [f"{r.name}[{r.level.value}]" for r in res if r.level != Level.PASS]
                row["recheck_nonpass"] = ";".join(nonpass)
                new_checks = {r.name: r.level.value for r in res}
            except Exception as exc:  # noqa: BLE001
                row["recheck"] = "ERROR"
                row["recheck_nonpass"] = repr(exc)[:150]
                new_checks = {}
        else:
            row["recheck"], row["recheck_nonpass"], new_checks = "NO_METRICS", "", {}
        old_checks = {c["name"]: c["level"] for c in (stored_rep or {}).get("checks", [])}
        row["stored"] = (stored_rep or {}).get("overall_verdict", "MISSING")
        row["verdict_differs"] = row["stored"] != row["recheck"]
        row["checks_differ"] = ";".join(
            f"{k}:{old_checks.get(k, '-')}->{new_checks.get(k, '-')}"
            for k in sorted(set(old_checks) | set(new_checks))
            if old_checks.get(k) != new_checks.get(k)
        )
        # 2. provenance
        git = meta.get("git", {})
        env = meta.get("environment", {})
        first, last = _log_window(p)
        row.update(
            {
                "git_commit": (git.get("commit") or "")[:10],
                "git_dirty": git.get("dirty"),
                "run_start_utc": first.strftime("%Y-%m-%d %H:%M:%S") if first else "",
                "run_end_utc": last.strftime("%Y-%m-%d %H:%M:%S") if last else "",
                "total_seconds": (meta.get("timing") or {}).get("total_seconds", ""),
                "torch": env.get("torch_version", ""),
                "gpu": env.get("gpu_name", ""),
                "has_run_metadata": bool(meta),
                "synced_mtime": datetime.fromtimestamp(
                    (p / ("events.json" if events is not None else "config.json")).stat().st_mtime,
                    tz=timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "uncommitted_fields_in_config": schema_info(cfg),
                "code_generation": code_generation(cfg),
                "needs_uncommitted_code": needs_uncommitted(cfg),
                "fix_flags": ";".join(fix_flags(cfg, first, kind)),
            }
        )
        # 3. config integrity
        row.update(config_compare(cfg, base, idx))
        row["ckpt_path"] = cfg.get("pre_deploy_checkpoint") if regime == "deployment" else ""
        row.update(fingerprint(events))
        pp = (cfg.get("retraining") or {}).get("priors_path")
        row["priors_path"] = pp or ""
        row["priors_file_available"] = (
            ""
            if not pp
            else bool((REPO / pp).exists() or (RESULTS / "_priors" / Path(pp).name).exists())
        )
        eff = next(
            (
                e.get("effective_params")
                for e in (events or {}).get("events", [])
                if e.get("effective_params")
            ),
            None,
        )
        row["effective_burst_lr"] = eff.get("burst_lr") if eff else ""
        row["effective_params_recorded"] = bool(eff)
        if events is not None:
            s = events.get("summary") or {}
            row["final_test_acc"] = s.get("final_test_acc")
            row["lifetime_mean"] = s.get("lifetime_mean_test_acc")
            row["burst_batches"] = s.get("total_burst_batches")
            row["n_waves_expected"] = len((cfg.get("fault") or {}).get("wave_densities") or [])
        row["_cfg"] = cfg
        row["_events"] = events
        rows.append(row)
    return rows, bad_yaml


def pair_defs():
    """Comparisons used in the report / audit_pairs.py: (label, armA, armB, liveA, liveB)."""
    p = []

    def add(label, a, b, la=None, lb=None):
        p.append((label, a, b, la, lb))

    add(
        "F4 D vs C'M (tuned)", "resnet9_deploy_d_eta3e-1", "resnet9_deploy_c_living_matched_eta5e-1"
    )
    add("F4 D vs BM (tuned)", "resnet9_deploy_d_eta3e-1", "resnet9_deploy_b_matched_eta5e-1")
    for ab in (
        "nopace_etaBD",
        "nopace_etaBD2",
        "nosal_etaBD",
        "nosal_etaBD2",
        "mag_etaBD",
        "mag_etaBD2",
    ):
        add(f"F5 D vs {ab}", "resnet9_deploy_d_eta3e-1", "resnet9_deploy_d_" + ab)
    for t in ("rf02", "rf05", "rf10", "d15", "d20", "d25", "w6", "w8"):
        add(
            f"F6 stress {t} D vs C'M",
            "resnet9_deploy_d_stress_" + t,
            "resnet9_deploy_c_living_stressM_" + t,
        )
        add(
            f"F6 stress {t} D vs BM",
            "resnet9_deploy_d_stress_" + t,
            "resnet9_deploy_b_stressM_" + t,
        )
    for w in ("w6", "w8"):
        add(
            f"gfix D vs C'M {w}",
            "resnet9_deploy_d_gfix_" + w,
            "resnet9_deploy_c_living_matchDgfix_" + w,
        )
    add("g5 gfix D vs C'M", "resnet9_deploy_d_g5_gfix", "resnet9_deploy_c_living_g5_matched", 6, 6)
    add("cont D vs C'M", "resnet9_cont_next_d", "resnet9_cont_next_c_living_matched")
    add(
        "c100 D vs C'M eta1",
        "resnet9_next_deploy_d_c100_eta0.3",
        "resnet9_next_deploy_c_living_c100_matched_eta1.0",
    )
    add(
        "c100 D vs C'M eta2",
        "resnet9_next_deploy_d_c100_eta0.3",
        "resnet9_next_deploy_c_living_c100_matched_eta2.0",
    )
    add("vgg6 D vs C'M", "vgg6_net2_deploy_d_best", "vgg6_net2_deploy_c_living_matched_best")
    add("vgg6 D vs BM", "vgg6_net2_deploy_d_best", "vgg6_net2_deploy_b_matched_best")
    add("vit D vs C'M", "vit_net3_deploy_d_best", "vit_net3_deploy_c_living_matched_best")
    add("vit D vs BM", "vit_net3_deploy_d_best", "vit_net3_deploy_b_matched_best")
    for s in (1, 2, 3):
        add(f"real S{s} D vs C'M", f"resnet9_real_s{s}_d", f"resnet9_real_s{s}_c_living_matched")
        add(f"real S{s} D vs BM", f"resnet9_real_s{s}_d", f"resnet9_real_s{s}_b_matched")
        add(f"real S{s} D vs R1", f"resnet9_real_s{s}_d", f"resnet9_real_s{s}_r1")
    for sc in ("s4a", "s4b", "s4a5", "s4b5"):
        add(f"S4 {sc} D vs BM", f"resnet9_calib_{sc}_d", f"resnet9_calib_{sc}_b_matched")
        add(f"S4 {sc} D2 vs D", f"resnet9_d2calib_{sc}_d2", f"resnet9_calib_{sc}_d")
        add(f"S4 {sc} D2 vs BM(d2)", f"resnet9_d2calib_{sc}_d2", f"resnet9_d2calib_{sc}_b_matched")
        add(f"S4 {sc} D2 vs C2", f"resnet9_d2calib_{sc}_d2", f"resnet9_d2calib_{sc}_c2")
    add("S4 s4b5 D vs C'M", "resnet9_calib_s4b5_d", "resnet9_calib_s4b5_c_living_matched")
    add("S4 s4b5 D2 vs C'M(d2)", "resnet9_d2calib_s4b5_d2", "resnet9_d2calib_s4b5_c_living_matched")
    add("D2 main D2 vs D(final)", "resnet9_d2_d2", "resnet9_final_d")
    add("D2 main D2 vs C'M", "resnet9_d2_d2", "resnet9_d2_c_living_matched")
    add("D2 main D2 vs BM", "resnet9_d2_d2", "resnet9_d2_b_matched")
    add("D2 main D2 vs C2", "resnet9_d2_d2", "resnet9_d2_c2")
    add("final D vs C'M", "resnet9_final_d", "resnet9_final_c_living_matched")
    add("final D vs BM", "resnet9_final_d", "resnet9_final_b_matched")
    add("final D vs R1", "resnet9_final_d", "resnet9_final_r1")
    add("vgg6 final D vs C'M", "vgg6_final_deploy_d", "vgg6_final_deploy_c_living_matched")
    add("vit final D vs C'M", "vit_final_deploy_d", "vit_final_deploy_c_living_matched")
    add("stage1 D vs C'", "resnet9_deploy_d", "resnet9_deploy_c_living")
    add("stage1 D vs BM", "resnet9_deploy_d", "resnet9_deploy_b_matched")
    add("stage1 D vs R1", "resnet9_deploy_d", "resnet9_deploy_r1")
    add("stage2 D vs noburst", "resnet9_deploy_d", "resnet9_deploy_d_noburst")
    add(
        "D2 main D2 vs D(tuned eta0.3, seeds 42/101/102)",
        "resnet9_d2_d2",
        "resnet9_deploy_d_eta3e-1",
    )
    add(
        "liu-style vs D (eta 0.3 sweep)",
        "resnet9_deploy_liu_style_lr0.3",
        "resnet9_deploy_d_eta3e-1",
    )
    return p


def pick(rows_by_key, base, seed, live):
    """Newest complete deployment/continuous row for (base, seed); returns (row, n_candidates)."""
    cands = [
        r
        for r in rows_by_key.get((base, seed), [])
        if (live is None or r["live"] == f"_live{live}")
        and r["has_metrics"]
        and r["kind"] != "predeploy"
    ]
    cands = [r for r in cands if r["complete"] in (True, 1)] or cands
    cands.sort(key=lambda r: r["run_start_utc"])
    return (cands[-1] if cands else None), len(cands)


def pair_rows(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["run"], r["seed"])].append(r)
    out = []
    for label, a, b, la, lb in pair_defs():
        seeds_a = {s for (n, s) in by if n == a}
        seeds_b = {s for (n, s) in by if n == b}
        if not seeds_a or not seeds_b:
            out.append(
                {
                    "pair": label,
                    "armA": a,
                    "armB": b,
                    "seed": "",
                    "status": "ARM_MISSING",
                    "detail": f"A seeds={sorted(seeds_a)} B seeds={sorted(seeds_b)}",
                }
            )
            continue
        for s in sorted(seeds_a & seeds_b):
            ra, na = pick(by, a, s, la)
            rb, nb = pick(by, b, s, lb)
            probs = []
            if ra["live"] != rb["live"]:
                probs.append(f"different_results_dirs({ra['live']} vs {rb['live']})")
            if ra["kind"] == "deployment" and rb["kind"] == "deployment":
                for k in ("ckpt_path", "ckpt_hash", "ckpt_acc", "base_test", "base_canary"):
                    if ra.get(k) != rb.get(k):
                        probs.append(f"{k}:{ra.get(k)}!={rb.get(k)}")
                if ra.get("densities") != rb.get("densities"):
                    probs.append("wave_densities_differ")
                ha, hb = json.loads(ra["wave_hashes"]), json.loads(rb["wave_hashes"])
                n = min(len(ha), len(hb))
                if ha[:n] != hb[:n]:
                    probs.append(
                        f"fault_map_hash_differs_waves={[i for i in range(n) if ha[i] != hb[i]]}"
                    )
                fa, fb = ra["_cfg"]["fault"], rb["_cfg"]["fault"]
                if fa != fb:
                    probs.append(
                        "fault_cfg_differs:" + ",".join(k for k in fa if fa[k] != fb.get(k))
                    )
                xa, xb = dict(ra["_cfg"]["crossbar"]), dict(rb["_cfg"]["crossbar"])
                if not (
                    ra["_cfg"]["mitigation"]["enable_remapping"]
                    and rb["_cfg"]["mitigation"]["enable_remapping"]
                ):
                    # a no-remap arm (B) never touches spares, so its redundancy setting is inert
                    xa.pop("redundancy_factor", None)
                    xb.pop("redundancy_factor", None)
                if xa != xb:
                    probs.append("crossbar_cfg_differs")
                if ra.get("gpu") != rb.get("gpu"):
                    probs.append(f"different_gpu({ra.get('gpu')} vs {rb.get('gpu')})")
                if ra["n_events"] != rb["n_events"]:
                    probs.append("n_waves_differ")
            else:
                if ra["kind"] != rb["kind"]:
                    probs.append("kind_differs")
                fa, fb = ra["_cfg"].get("fault"), rb["_cfg"].get("fault")
                if fa != fb:
                    probs.append("fault_cfg_differs")
            if na > 1 or nb > 1:
                probs.append(f"duplicates_present(A={na},B={nb})")
            out.append(
                {
                    "pair": label,
                    "armA": a,
                    "armB": b,
                    "seed": s,
                    "dirA": ra["live"],
                    "dirB": rb["live"],
                    "stepsA": ra.get("burst_batches"),
                    "stepsB": rb.get("burst_batches"),
                    "finalA": ra.get("final_test_acc"),
                    "finalB": rb.get("final_test_acc"),
                    "gpuA": ra.get("gpu"),
                    "gpuB": rb.get("gpu"),
                    "status": "OK" if not probs else "FLAG",
                    "detail": " | ".join(probs),
                }
            )
    return out


def dup_rows(rows):
    by = defaultdict(list)
    for r in rows:
        if r["kind"] != "predeploy" and r["has_metrics"]:
            by[(r["run"], r["seed"])].append(r)
    out = []
    for (run, seed), rs in sorted(by.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: (r["run_start_utc"], r["live"]))
        sig = lambda r: (
            r.get("final_test_acc"),
            r.get("lifetime_mean"),
            r.get("wave_hashes"),
            r.get("burst_batches"),
        )
        same = len({sig(r) for r in rs}) == 1
        cfgs = {json.dumps(r["_cfg"], sort_keys=True) for r in rs}
        out.append(
            {
                "run": run,
                "seed": seed,
                "dirs": ",".join(r["live"] for r in rs),
                "complete": ",".join(str(r["complete"]) for r in rs),
                "identical_results": same,
                "identical_config": len(cfgs) == 1,
                "finals": ",".join(str(r.get("final_test_acc")) for r in rs),
                "n_events": ",".join(str(r.get("n_events")) for r in rs),
                "starts": ",".join(r["run_start_utc"] for r in rs),
                "newest": rs[-1]["live"],
            }
        )
    return out


def predeploy_shas(rows):
    out = []
    for r in rows:
        if r["kind"] != "predeploy":
            continue
        f = REPO / r["path"] / "pre_deploy.pt"
        sha = hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.exists() else ""
        out.append(
            {
                "path": r["path"],
                "run": r["run"],
                "seed": r["seed"],
                "live": r["live"],
                "sha12": sha,
                "start": r["run_start_utc"],
                "gpu": r["gpu"],
            }
        )
    return out


def hash_buckets(rows):
    """Do identical (checkpoint, seed, fault config) runs produce identical fault maps? Split by GPU."""
    b = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["kind"] != "deployment" or r["complete"] is not True:
            continue
        c = r["_cfg"]
        key = (
            r["arch"],
            r["dataset"],
            r["seed"],
            r["ckpt_hash"],
            r["ckpt_acc"],
            json.dumps(c["fault"], sort_keys=True),
            json.dumps(c.get("secondary_fault"), sort_keys=True),
            c["crossbar"].get("scheme"),
            json.dumps(c["crossbar"].get("tile_shape")),
        )
        b[key][r["gpu"]].add(json.loads(r["wave_hashes"])[0])
    n_multi_gpu = same_gpu_conflict = gpu_dependent = 0
    base_conflict = 0
    for per_gpu in b.values():
        if any(len(v) > 1 for v in per_gpu.values()):
            same_gpu_conflict += 1
        if len(per_gpu) > 1:
            n_multi_gpu += 1
            if len(set().union(*per_gpu.values())) > 1:
                gpu_dependent += 1
    by_ckpt = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["kind"] == "deployment" and r["complete"] is True:
            by_ckpt[(r["seed"], r["ckpt_hash"], r["ckpt_acc"], r["arch"], r["dataset"])][
                r["gpu"]
            ].add(r["base_test"])
    for per_gpu in by_ckpt.values():
        if len(set().union(*per_gpu.values())) > 1:
            base_conflict += 1
    return {
        "buckets": len(b),
        "buckets_with_runs_on_multiple_gpus": n_multi_gpu,
        "multi_gpu_buckets_with_different_wave0_maps": gpu_dependent,
        "buckets_with_differing_maps_on_SAME_gpu": same_gpu_conflict,
        "ckpt_groups_with_differing_clean_test_acc_across_gpus_or_runs": base_conflict,
    }


def write_csv(path: Path, rows, drop=("_cfg", "_events")):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and k not in drop:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def group_rows(rows, pairs):
    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    out = []
    for name, rs in sorted(g.items()):
        dep = [r for r in rs if r["kind"] == "deployment"]
        out.append(
            {
                "group": name,
                "runs": len(rs),
                "deployment_complete": sum(1 for r in dep if r["complete"] is True),
                "deployment_incomplete": sum(1 for r in dep if r["complete"] is not True),
                "recheck_FAIL": sum(r["recheck"] == "FAIL" for r in rs),
                "recheck_WARN": sum(r["recheck"] == "WARN" for r in rs),
                "verdict_differs": sum(bool(r["verdict_differs"]) for r in rs),
                "no_git": sum(not r["git_commit"] for r in rs),
                "cfg_differs": sum(r["cfg_status"] == "differs" for r in rs),
                "cfg_no_yaml": sum(r["cfg_status"] == "no_yaml" for r in rs),
                "fix_flagged": sum(bool(r["fix_flags"]) for r in rs),
                "needs_uncommitted_code": sum(bool(r["uncommitted_fields_in_config"]) for r in rs),
                "dirs": ",".join(sorted({r["live"] for r in rs})),
            }
        )
    return out


_RAW_NEEDED = (
    "needs the raw run archive (per-run run.log, checkpoint .pt and validation_report.json files), "
    "which is not shipped in this repository. Its outputs are shipped instead under "
    "results/derived/ (pair_audit.csv, dup_audit.csv, predeploy_checkpoints.csv). Nothing was run and nothing was written."
)


def _raw_archive_present() -> bool:
    """True when per-run run.log files exist under results/ (not shipped in the public repo)."""
    return any(RESULTS.glob("*/run.log")) or any(RESULTS.glob("*/*/run.log"))


def main() -> None:
    if not _raw_archive_present():
        print("audit_runs.py " + _RAW_NEEDED)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows, bad_yaml = audit_rows()
    pairs = pair_rows(rows)
    dups = dup_rows(rows)
    ckpts = predeploy_shas(rows)
    write_csv(OUT / "run_audit.csv", rows)
    write_csv(OUT / "pair_audit.csv", pairs)
    write_csv(OUT / "dup_audit.csv", dups)
    write_csv(OUT / "predeploy_checkpoints.csv", ckpts)
    write_csv(OUT / "group_audit.csv", group_rows(rows, pairs))
    dep = [r for r in rows if r["kind"] == "deployment"]
    summary = {
        "runs_total": len(rows),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "deployment_complete": sum(r["complete"] is True for r in dep),
        "deployment_incomplete": [r["path"] for r in dep if r["complete"] is not True],
        "recheck": dict(Counter(r["recheck"] for r in rows)),
        "recheck_deployment_complete": dict(
            Counter(r["recheck"] for r in dep if r["complete"] is True)
        ),
        "stored": dict(Counter(r["stored"] for r in rows)),
        "verdict_differs": [r["path"] for r in rows if r["verdict_differs"]],
        "non_pass": [
            (r["path"], r["recheck"], r["recheck_nonpass"])
            for r in rows
            if r["recheck"] not in ("PASS",)
        ],
        "git_recorded": sum(bool(r["git_commit"]) for r in rows),
        "git_dirty": sum(bool(r["git_commit"]) and r["git_dirty"] is True for r in rows),
        "cfg_status": dict(Counter(r["cfg_status"] for r in rows)),
        "pairs": dict(Counter(p["status"] for p in pairs)),
        "dups": len(dups),
        "dups_not_identical": [
            d["run"] + f"_seed{d['seed']}" for d in dups if not d["identical_results"]
        ],
        "bad_yaml": bad_yaml,
        "hash_buckets": hash_buckets(rows),
        "priors_missing": sum(r["priors_file_available"] is False for r in rows),
        "priors_missing_but_effective_recorded": sum(
            r["priors_file_available"] is False and r["effective_params_recorded"] for r in rows
        ),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not isinstance(v, list)}, indent=1, default=str
        )
    )


if __name__ == "__main__":
    main()

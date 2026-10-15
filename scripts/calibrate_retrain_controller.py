"""Calibration of the co-mitigation retraining controller (plan: "Calibration").

Two modes, with deliberately different authority:

  --from-past-results <results_dir>
      Reads results/resnet9_fault_{no_mitigation,remapping,layer_reset,combined}
      _seed42/metrics.json, detects fault-wave batches from jumps in the
      `faults` list, fits acc(t) = A_inf - (A_inf - A0) * exp(-t/tau) per wave
      segment (wave batch to next wave, or end of run), and prints a table.
      These are TEST-accuracy curves, so the output is ONLY search ranges
      (T_max range, patience window m*k range, A_inf-vs-stuck-fraction points).
      Per the plan they must NOT be used to tune thresholds (delta, kappa, ...);
      doing so would leak the test set into the controller.

  --from-calibration-sweep <dir>
      Reads <dir>/canary_curves.json (format below), fits an exponential per
      eta, simulates the controller offline, grid-searches (eta, delta, kappa,
      k, m, T_max, pacing) and writes experiments/configs/
      retrain_controller_priors.json. Thresholds are selected here, on CANARY
      accuracy only.

canary_curves.json format (written by experiments/calibration_sweep.py):
  {
    "metadata": {"n_canary": 1024, "clean_acc": 89.8,   # clean_acc = pre-wave CANARY acc
                 "seed": 42, "checkpoint": "<pre-deploy checkpoint id>"},
    "curves": {"0.1": {"0.0001": [[0, 55.1], [25, 58.0], ...],   # fraction -> eta -> curve
                       "0.001": [...], "0.005": [...]},
               "0.05": {"0.001": [...]}, "0.2": {"0.001": [...]}}
  }
Batch 0 is the canary right after the wave. Legacy single-fraction files
({"metadata": {..., "wave_fraction": 0.1}, "curves": {"<eta>": [[b, acc], ...]}}) are
still accepted and treated as {"<wave_fraction>": curves}.

Recovery model (per eta): tau from the reference fraction's curve (the fraction measured
at the most etas, ties -> closest to 0.10); A_inf(u) and A_post(u) are MEASURED at every
fraction u with a curve (A_post = the batch-0 canary, A_inf = the exponential fit's
asymptote, clipped to [A_post, clean]). Where an eta lacks a curve at some fraction the
point is borrowed from the best-measured eta at that fraction (flagged
"borrowed_from_eta"); beyond the largest measured fraction the model extends linearly
(flagged "extrapolated") out to the simulated lifetime's cumulative stuck fraction. The
result inherits the single-seed, few-fractions caveat. Only canary accuracy is used.

Usage:
    uv run scripts/calibrate_retrain_controller.py --from-past-results results/
    uv run scripts/calibrate_retrain_controller.py --from-calibration-sweep <dir> --lam 0.002
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neurofault.provenance import capture_git_info
from neurofault.retraining.recovery_model import (
    RecoveryModel,
    Wave,
    fit_exponential_recovery,
    grid_search,
)

logger = logging.getLogger("calibrate_retrain_controller")

ARMS = ("no_mitigation", "remapping", "layer_reset", "combined")
PRIORS_PATH = Path("experiments/configs/retrain_controller_priors.json")


def detect_waves(batches: list[int], faults: list[float]) -> list[int]:
    """Indices into `batches` at which the cumulative fault count jumps."""
    return [i for i in range(1, len(faults)) if faults[i] > faults[i - 1]]


def fit_wave_segments(metrics: dict, density: float | None = None) -> list[dict]:
    """Per-wave fits. Cumulative stuck fraction is estimated as
    cumulative faults / (first-wave jump / density) when `density` is given."""
    batches, acc, faults = metrics["batch"], metrics["accuracy"], metrics["faults"]
    idx = detect_waves(batches, faults)
    rows = []
    for j, i in enumerate(idx):
        end = idx[j + 1] if j + 1 < len(idx) else len(batches)
        t = np.array(batches[i:end], dtype=float) - batches[i]
        fit = fit_exponential_recovery(t, acc[i:end])
        total_cells = (faults[idx[0]] - faults[idx[0] - 1]) / density if density else None
        rows.append(
            {
                "wave": j + 1,
                "batch": batches[i],
                "n_points": end - i,
                "stuck_fraction": faults[i] / total_cells if total_cells else None,
                "fit": fit,
            }
        )
    return rows


def run_past_results(results_dir: Path) -> int:
    print("PAST-RESULTS MODE: test-accuracy curves -> SEARCH RANGES ONLY.")
    print("Do NOT tune thresholds (delta, kappa, ...) on these; see script docstring.\n")
    header = f"{'arm':<14}{'wave':>5}{'batch':>8}{'tau':>9}{'A_inf':>8}{'A0':>8}{'rmse':>7}  ok"
    print(header)
    taus, no_mit_points = [], []
    for arm in ARMS:
        run = results_dir / f"resnet9_fault_{arm}_seed42"
        mpath = run / "metrics.json"
        if not mpath.exists():
            print(f"{arm:<14} (missing {mpath})")
            continue
        metrics = json.loads(mpath.read_text())
        density = json.loads((run / "config.json").read_text()).get("fault", {}).get("density")
        for r in fit_wave_segments(metrics, density):
            f = r["fit"]
            print(
                f"{arm:<14}{r['wave']:>5}{r['batch']:>8}{f.tau:>9.0f}{f.a_inf:>8.1f}"
                f"{f.a0:>8.1f}{f.rmse:>7.2f}  {'y' if f.converged else 'FALLBACK'}"
            )
            # Ranges come from the no_mitigation arm only: remap-bearing arms'
            # wave 1 is absorbed almost instantly (tau ~15-18), which says
            # nothing about how long a retraining burst needs.
            if arm == "no_mitigation" and f.converged:
                taus.append(f.tau)
            if arm == "no_mitigation" and r["stuck_fraction"] is not None:
                no_mit_points.append((r["stuck_fraction"], f.a_inf))
    if not taus:
        print("no fits; nothing to report")
        return 1
    lo, hi = min(taus), max(taus)
    print("\nSEARCH RANGES (not tuned values):")
    print(
        f"  T_max range (batches):        [{0.5 * lo:.0f}, {2.0 * hi:.0f}]  (0.5*min tau .. 2*max tau, no_mitigation)"
    )
    print(f"  patience window m*k (batches): [{lo / 20:.0f}, {lo / 4:.0f}]  (min tau / 20 .. / 4)")
    if no_mit_points:
        pts = ", ".join(f"({u:.2f}, {a:.1f})" for u, a in no_mit_points)
        print(f"  A_inf vs stuck fraction (no_mitigation, test acc): {pts}")
    return 0


def normalize_curves(data: dict) -> dict[float, dict[float, list]]:
    """{fraction: {eta: [[batch, acc], ...]}} from either file format."""
    curves = data["curves"]
    if all(isinstance(v, dict) for v in curves.values()):
        return {float(f): {float(e): c for e, c in per.items()} for f, per in curves.items()}
    return {float(data["metadata"]["wave_fraction"]): {float(e): c for e, c in curves.items()}}


def build_recovery_models(
    data: dict, u_end: float
) -> tuple[dict[float, RecoveryModel], dict, float, dict]:
    """Per-eta RecoveryModels from measured canary curves (see module docstring).
    Returns (models, fitted per (fraction, eta), reference fraction, point provenance)."""
    clean = float(data["metadata"]["clean_acc"])
    by_f = normalize_curves(data)
    fitted: dict[tuple[float, float], dict] = {}
    for f, per in by_f.items():
        for eta, curve in per.items():
            arr = np.asarray(curve, dtype=float)
            fit = fit_exponential_recovery(arr[:, 0] - arr[0, 0], arr[:, 1])
            fitted[(f, eta)] = {
                "tau": fit.tau,
                "a_inf": fit.a_inf,
                "a0": fit.a0,
                "a_post": float(arr[0, 1]),
                "rmse": fit.rmse,
                "converged": fit.converged,
            }
    ref_f = min(by_f, key=lambda f: (-len(by_f[f]), abs(f - 0.1)))
    etas = sorted(by_f[ref_f])
    models: dict[float, RecoveryModel] = {}
    provenance: dict[str, dict] = {}
    for eta in etas:
        pts: dict[float, tuple[float, float, str]] = {0.0: (clean, clean, "clean")}
        for f in sorted(by_f):
            src = eta if eta in by_f[f] else None
            how = "measured"
            if src is None:  # borrow from the eta with the most similar learning rate
                src = min(by_f[f], key=lambda e: abs(np.log(e / eta)))
                how = f"borrowed_from_eta={src:g}"
            ft = fitted[(f, src)]
            a_post = min(ft["a_post"], clean)
            a_inf = min(max(ft["a_inf"], a_post), clean)
            pts[f] = (a_inf, a_post, how)
        fs = sorted(pts)
        a_inf = list(np.minimum.accumulate([pts[f][0] for f in fs]))
        a_post = list(np.minimum.accumulate([pts[f][1] for f in fs]))
        if u_end > fs[-1] and len(fs) >= 2:
            # linear continuation of the last measured segment (flagged: an assumption)
            for series in (a_inf, a_post):
                slope = (series[-1] - series[-2]) / (fs[-1] - fs[-2])
                series.append(max(series[-1] + slope * (u_end - fs[-1]), 10.0))
            fs.append(u_end)
            pts[u_end] = (a_inf[-1], a_post[-1], "extrapolated")
        models[eta] = RecoveryModel(
            eta=eta,
            tau=fitted[(ref_f, eta)]["tau"],
            stuck_fractions=fs,
            a_inf=[float(x) for x in a_inf],
            a_post=[float(x) for x in a_post],
        )
        provenance[repr(eta)] = {repr(f): pts[f][2] for f in sorted(pts)}
    return models, fitted, ref_f, provenance


def run_calibration_sweep(sweep_dir: Path, args: argparse.Namespace) -> int:
    data = json.loads((sweep_dir / "canary_curves.json").read_text())
    meta = data["metadata"]
    clean = float(meta["clean_acc"])
    by_f = normalize_curves(data)
    f_ref = args.wave_fraction or min(by_f, key=lambda f: (-len(by_f[f]), abs(f - 0.1)))
    models, fitted, ref_f, point_source = build_recovery_models(data, args.n_waves * f_ref)
    for eta, m in models.items():
        ft = fitted[(ref_f, eta)]
        print(
            f"eta={eta:g}: tau={ft['tau']:.0f} A_inf={ft['a_inf']:.1f} A0={ft['a0']:.1f} "
            f"rmse={ft['rmse']:.2f} converged={ft['converged']} (fraction {ref_f:g}); "
            f"u={[round(u, 3) for u in m.stuck_fractions]} A_inf={[round(a, 1) for a in m.a_inf]} "
            f"A_post={[round(a, 1) for a in m.a_post]}"
        )
    f1 = f_ref
    waves = [Wave(args.first_wave + i * args.wave_gap, f1) for i in range(args.n_waves)]
    horizon = waves[-1].batch + args.wave_gap
    result = grid_search(
        models,
        waves,
        args.capacity,
        lam=args.lam,
        horizon=horizon,
        clean_acc=clean,
        n_canary=int(meta.get("n_canary", 1000)),
        seed=args.seed,
    )
    print(f"\nPareto set ({len(result.pareto)} of {len(result.points)} points), lam={args.lam}:")
    for p in result.pareto:
        q = p.params
        print(
            f"  mean={p.result.lifetime_mean_acc:6.2f} burst={p.result.burst_batches:7.0f}  "
            f"eta={q.eta:g} delta={q.delta} kappa={q.kappa} k={q.k} m={q.m} "
            f"T_max={q.t_max} {q.pacing}"
        )
    c = result.chosen.params
    n = len(waves)
    priors = {
        "eta": c.eta,
        "delta": c.delta,
        "kappa": c.kappa,
        "k": c.k,
        "m": c.m,
        "t_max": c.t_max,
        "rho_schedule": ([1.0 / (n - i) for i in range(n)] if c.pacing == "paced" else [1.0] * n),
        "pacing": c.pacing,
        "lam": args.lam,
        "chosen_lifetime_mean_acc": result.chosen.result.lifetime_mean_acc,
        "chosen_burst_batches": result.chosen.result.burst_batches,
        "fitted_per_eta": {repr(e): fitted[(ref_f, e)] for e in models},  # at ref fraction
        "fitted_per_fraction_eta": {f"{f!r}/{e!r}": v for (f, e), v in fitted.items()},
        "model_points": {
            repr(eta): {
                "stuck_fractions": m.stuck_fractions,
                "a_inf": m.a_inf,
                "a_post": m.a_post,
                "tau": m.tau,
                "source": point_source[repr(eta)],
            }
            for eta, m in models.items()
        },
        "reference_fraction": ref_f,
        "provenance": {
            "inputs": str(sweep_dir / "canary_curves.json"),
            "input_metadata": meta,
            "date": datetime.now(tz=timezone.utc).date().isoformat(),
            "git": capture_git_info(),
            "simulated_waves": [[w.batch, w.new_fault_fraction] for w in waves],
            "remap_capacity": args.capacity,
            "caveat": "one seed's checkpoint, one wave's canary recovery curves",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=2))
    print(f"\nwrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-past-results", type=Path, metavar="RESULTS_DIR")
    g.add_argument("--from-calibration-sweep", type=Path, metavar="DIR")
    ap.add_argument("--lam", type=float, default=0.002, help="accuracy points per burst batch")
    ap.add_argument("--capacity", type=float, default=0.15, help="spares, fraction of cells")
    ap.add_argument(
        "--wave-fraction",
        type=float,
        default=None,
        help="simulated new-fault fraction per wave (default: the reference fraction)",
    )
    ap.add_argument("--n-waves", type=int, default=3)
    ap.add_argument("--first-wave", type=int, default=5200)
    ap.add_argument("--wave-gap", type=int, default=5200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PRIORS_PATH))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.from_past_results:
        return run_past_results(args.from_past_results)
    return run_calibration_sweep(args.from_calibration_sweep, args)


if __name__ == "__main__":
    sys.exit(main())

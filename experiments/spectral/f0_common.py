"""Shared plumbing for the F0 falsification scripts.

Why one module: F0.0 and F0.1 must measure the old feasibility probe the same way (F0.0a
reproduces its table, F0.1 H1.6 re-uses it as a baseline), and both end the same way: write the
provenance-stamped run directory, then let `spectral.verdict` turn metrics.csv into verdict.json
against the FROZEN criteria file. Nothing here judges a number.

Config files are YAML with two top-level blocks, `full:` and `smoke:`; `--smoke` selects the
small one. Smoke runs never count (their output directory gets a `_smoke` suffix).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTRAL_DIR = Path(__file__).resolve().parent
CRITERIA_DIR = SPECTRAL_DIR / "criteria"
CONFIG_DIR = SPECTRAL_DIR / "configs"

from neurofault.spectral import metrics as M
from neurofault.spectral.encoding import encode
from neurofault.spectral.faults import inject
from neurofault.spectral.harness import write_run
from neurofault.spectral.params import dump_params
from neurofault.spectral.readout import TileReader
from neurofault.spectral.tiles import checkpoint_path, load_checkpoint
from neurofault.spectral.types import (
    DTYPE,
    EncodingSpec,
    FaultSpec,
    ReadNoise,
    Tile,
    TileSpec,
    make_generator,
)
from neurofault.spectral.verdict import evaluate, load_criteria, write_verdict

# the four layers of the original feasibility probe
PROBE_LAYERS = [
    "layer1.0.0.weight",
    "res1.conv_a.weight",
    "layer3.0.0.weight",
    "res2.conv_b.weight",
]

_CKPT_CACHE: dict[int, dict[str, torch.Tensor]] = {}


# --------------------------------------------------------------------------- CLI / config


def parse_args(description: str, default_config: str, default_out: str, argv=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(CONFIG_DIR / default_config))
    ap.add_argument("--seed", type=int, default=0, help="master seed for all RNG streams")
    default_path = str(REPO_ROOT / "results" / "spectral" / default_out)
    ap.add_argument("--out", default=default_path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="small grid; never counts as a result")
    a = ap.parse_args(argv)
    if a.smoke and a.out == default_path:
        a.out = default_path + "_smoke"
    return a


def load_config(path: str | Path, smoke: bool) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = raw["smoke" if smoke else "full"]
    cfg = dict(cfg)
    cfg["_variant"] = "smoke" if smoke else "full"
    return cfg


def finish(
    exp: str,
    criteria_file: str,
    cfg: dict,
    args,
    metrics: pd.DataFrame,
    extra: dict | None = None,
    side_tables: dict[str, pd.DataFrame] | None = None,
) -> tuple[Path, dict]:
    """Write the run directory and verdict.json; print the per-criterion verdict."""
    crit = load_criteria(CRITERIA_DIR / criteria_file)
    extra = dict(extra or {})
    extra.update(
        criteria_file=criteria_file,
        criteria_sha256=crit.sha256,
        variant=cfg.get("_variant"),
        jobs=args.jobs,
        checkpoint_sha256={
            str(s): hashlib.sha256(checkpoint_path(s).read_bytes()).hexdigest()
            for s in cfg.get("ckpt_seeds", [])
            if checkpoint_path(s).exists()
        },
    )
    run_dir = write_run(
        Path(args.out),
        cfg,
        args.seed,
        metrics,
        dump_params(),
        extra,
        allow_dirty=args.allow_dirty,
    )
    for name, df in (side_tables or {}).items():
        df.to_csv(run_dir / f"{name}.csv", index=False)
    verdict = evaluate(crit, metrics)
    write_verdict(verdict, run_dir / "verdict.json", extra={"variant": cfg.get("_variant")})
    print(f"\n=== {exp} verdict: {verdict['overall']}   ({run_dir}) ===")
    for c in verdict["criteria"]:
        vals = _leaf_values(c.get("detail", {}))
        print(f"  {c['id']:<14s} {c['kind']:<8s} {c['status']:<13s} {vals}")
    sys.stdout.flush()
    return run_dir, verdict


def _leaf_values(detail: dict) -> str:
    out: list[str] = []

    def walk(d):
        if "clauses" in d:
            for c in d["clauses"]:
                walk(c)
        elif "metric" in d:
            v = d.get("value")
            out.append(
                f"{d['metric']}[{d['aggregation']}]="
                f"{'n/a' if v is None else format(v, '.4g')}{d['comparator']}{d['threshold']}"
            )

    walk(detail)
    return "; ".join(out)


# --------------------------------------------------------------------------- tiles


def ckpt_mats(seed: int) -> dict[str, torch.Tensor]:
    if seed not in _CKPT_CACHE:
        _CKPT_CACHE[seed] = load_checkpoint(seed)
    return _CKPT_CACHE[seed]


def get_tile(ckpt_seed: int, layer: str, r0: int, c0: int, m: int, n: int, size: int) -> Tile:
    w = ckpt_mats(ckpt_seed)[layer][r0 : r0 + m, c0 : c0 + n].clone()
    return Tile(TileSpec(ckpt_seed, layer, r0, c0, m, n, size), w)


def enc_from_label(label: str) -> EncodingSpec:
    """'signed_ideal' | 'diff_pair_g0' | 'diff_pair_g0.01' | 'diff_pair_g0.05'."""
    if label == "signed_ideal":
        return EncodingSpec("signed_ideal")
    if label.startswith("diff_pair_g"):
        return EncodingSpec("diff_pair", g_min_frac=float(label[len("diff_pair_g") :]))
    raise ValueError(label)


# --------------------------------------------------------------------------- old probe


def sigma1(w: torch.Tensor) -> float:
    return float(torch.linalg.svdvals(w)[0])


def run_old_probe(
    ckpt_seed: int,
    layer: str,
    seed: int,
    *,
    T: int = 256,
    trials: int = 200,
    noise_frac: float = 0.02,
    n_noise_draws: int = 100,
    p_multi: int = 8,
) -> dict:
    """The original feasibility probe (signed_ideal, first TxT slice, per-cell read noise,
    one stuck-at-+-rail cell per trial) re-implemented on the new stack: encode / inject /
    TileReader. Returns the probe's table columns plus P=`p_multi` row localization.

    Definitions: sigma_1 SNR = |noise-free shift| / std over
    `n_noise_draws` noisy readouts; probe SNR = |dw| / (sigma_cell * sqrt(n)); row-loc =
    argmax of |(A_faulty + noise) x - A x| equals the faulty row, x a fresh +-1 vector per trial.
    """
    tile = get_tile(ckpt_seed, layer, 0, 0, T, T, T)
    enc = EncodingSpec("signed_ideal")
    st0 = encode(tile, enc)
    A = st0.weights()
    n = A.shape[1]
    wmax = float(A.abs().max())
    sn = noise_frac * wmax
    noise = ReadNoise("per_cell", noise_frac, adc_bits=None, dac_bits=None)
    gn = make_generator(seed, f"old_probe:{ckpt_seed}:{layer}:noise")
    gp = make_generator(seed, f"old_probe:{ckpt_seed}:{layer}:probe")
    gf = make_generator(seed, f"old_probe:{ckpt_seed}:{layer}:fault")

    S = torch.linalg.svdvals(A)
    s1, gap = float(S[0]), float(S[0] / S[1])
    rd = TileReader(lambda: st0, noise, gn)
    noise_s1 = torch.tensor([sigma1(rd.full_readout()) for _ in range(n_noise_draws)]).std()
    spec = FaultSpec("stuck_random", k=1, rail="lrs")
    eig_snr, prb_snr, det, loc1, locp = [], [], 0, 0, 0
    for _ in range(trials):
        stf, truth = inject(st0, spec, gf)
        i = int(truth.cells[0, 0])
        d = sigma1(stf.weights()) - s1
        eig_snr.append(abs(d) / float(noise_s1))
        det += abs(d) > 3 * float(noise_s1)
        prb_snr.append(float(truth.delta_w[0].abs()) / (sn * n**0.5))
        X = torch.sign(torch.randn(n, p_multi, generator=gp, dtype=DTYPE))
        X[X == 0] = 1.0
        rf = TileReader(lambda s=stf: s, noise, gn)
        R = rf.mvm(X) - A @ X
        loc1 += int(R[:, 0].abs().argmax()) == i
        locp += int((R**2).sum(1).argmax()) == i
    med = lambda v: float(torch.tensor(v).median())
    return {
        "gap": gap,
        "sigma1_snr_med": med(eig_snr),
        "sigma1_det3": det / trials,
        "probe_snr_med": med(prb_snr),
        "rowloc_p1": loc1 / trials,
        f"rowloc_p{p_multi}": locp / trials,
        "trials": trials,
    }


__all__ = ["FaultSpec", "M", "ReadNoise", "TileReader", "encode", "inject", "make_generator"]

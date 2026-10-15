"""Grid runner and provenance writer for spectral experiments.

Why: every number must be traceable to config + seed + commit, and a headline run on a dirty
tree is not reproducible. `write_run` therefore refuses a dirty tree unless allow_dirty=True
(and records that choice). `run_grid` is restartable: finished cells are appended to
`<out_dir>/cells.jsonl` keyed by a hash of the cell, so a crashed multi-hour F0.1 resumes.

Workers are single-threaded (OMP/MKL/torch threads = 1) so N processes use N cores instead of
oversubscribing; `fn` must be a picklable top-level function taking one dict and returning a
dict (one metrics row) or a list of dicts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from neurofault.provenance import capture_environment, capture_git_info
from neurofault.spectral.types import (
    EncodingSpec,
    FaultSpec,
    ProgNoise,
    ReadNoise,
    TileSpec,
)

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class DirtyTreeError(RuntimeError):
    """Raised when a run would be recorded against an uncommitted tree."""


@dataclass
class Scenario:
    """One cell of an experiment grid. `drift` is any DriftModel (kept untyped here so this
    module does not depend on drift.py being built)."""

    tile: TileSpec
    enc: EncodingSpec
    prog: ProgNoise
    read: ReadNoise
    drift: object | None
    fault: FaultSpec | None
    t: float
    seed: int

    def as_dict(self) -> dict:
        return _jsonable(self)


def _jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        d["__type__"] = type(obj).__name__
        return d
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "name"):  # DriftModel-like objects
        return {"__type__": type(obj).__name__, "name": obj.name}
    return repr(obj)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(_jsonable(cfg), sort_keys=True).encode()).hexdigest()


def _cell_key(point: dict) -> str:
    return config_hash(point)[:20]


def _init_worker() -> None:
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    import torch

    torch.set_num_threads(1)


def _call(args: tuple[Callable, dict]) -> list[dict]:
    fn, point = args
    res = fn(point)
    rows = [res] if isinstance(res, dict) else list(res)
    return [{**point_scalars(point), **r} for r in rows]


def point_scalars(point: dict) -> dict:
    """Grid keys that are plain scalars are echoed into every result row for later filtering."""
    return {k: v for k, v in point.items() if isinstance(v, (bool, int, float, str))}


def run_grid(
    fn: Callable[[dict], dict | list[dict]],
    grid: Iterable[dict],
    *,
    jobs: int = 1,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Run fn over every grid point (jobs processes) and return one DataFrame, grid order.

    With out_dir, rows are appended incrementally to out_dir/cells.jsonl and cells already
    present (same point hash) are skipped, so the run resumes after a crash."""
    points = list(grid)
    cache: dict[str, list[dict]] = {}
    cache_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "cells.jsonl"
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["rows"]
    todo = [(i, p) for i, p in enumerate(points) if _cell_key(p) not in cache]
    results: dict[int, list[dict]] = {}

    def _store(i: int, rows: list[dict]) -> None:
        results[i] = rows
        if cache_path is not None:
            with cache_path.open("a") as fh:
                fh.write(json.dumps({"key": _cell_key(points[i]), "rows": _jsonable(rows)}) + "\n")

    if jobs <= 1 or len(todo) <= 1:
        for i, p in todo:
            _store(i, _call((fn, p)))
    else:
        saved = {v: os.environ.get(v) for v in _THREAD_VARS}
        for v in _THREAD_VARS:
            os.environ[v] = "1"  # inherited by spawned children before torch loads
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(jobs, initializer=_init_worker) as pool:
                for (i, _), rows in zip(
                    todo, pool.imap(_call, [(fn, p) for _, p in todo], chunksize=1), strict=True
                ):
                    _store(i, rows)
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
    all_rows: list[dict] = []
    for i, p in enumerate(points):
        all_rows.extend(results[i] if i in results else cache[_cell_key(p)])
    return pd.DataFrame(all_rows)


def write_run(
    out_dir: Path,
    cfg: dict,
    seed: int,
    metrics: pd.DataFrame,
    params: dict,
    extra: dict | None = None,
    *,
    allow_dirty: bool = False,
    run_id: str | None = None,
) -> Path:
    """Write results/spectral/<exp>/<run_id>/{config,run_metadata,params}.json + metrics.csv.

    out_dir is the experiment directory (results/spectral/<exp>); the run directory is created
    inside it. Refuses a dirty tree unless allow_dirty (recorded in run_metadata.json)."""
    git = capture_git_info()
    if git.get("dirty") and not allow_dirty:
        raise DirtyTreeError(
            "working tree is dirty; commit first or pass allow_dirty=True (recorded in metadata)"
        )
    chash = config_hash({"cfg": cfg, "seed": seed})
    rid = run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{chash[:8]}_s{seed}"
    run_dir = Path(out_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(_jsonable(cfg), indent=2, sort_keys=True))
    meta = {
        "run_id": rid,
        "seed": seed,
        "config_hash": chash,
        "git": git,
        "allow_dirty": allow_dirty,
        "environment": capture_environment(),
        "argv": list(sys.argv),
        "written_at_unix": time.time(),
        "extra": _jsonable(extra or {}),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (run_dir / "params.json").write_text(json.dumps(_jsonable(params), indent=2, sort_keys=True))
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    return run_dir

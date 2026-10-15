"""Per-run provenance capture: which exact code and environment produced a
result. Written to `run_metadata.json` by experiments/run.py, kept separate
from config.json since this is runtime fact, not user-authored experiment
config. Every capture function is best-effort (try/except around anything
that can be absent - no git, no GPU, no optional backend installed) so a run
never fails just because provenance can't be fully captured."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch

# Best-effort: env var names that container/cloud hosts commonly use to expose an instance or
# container id. None were verified against a live instance during development - these are
# plausible candidates. Absent vars are silently skipped, never treated as an error; the first
# one present is recorded as `instance_id`.
_INSTANCE_ID_ENV_VARS = (
    "CONTAINER_ID",
    "VAST_CONTAINERLABEL",
    "VAST_CONTAINER_ID",
    "RUNPOD_POD_ID",
)


def capture_git_info() -> dict:
    """Commit hash, branch, and whether the working tree was dirty at run
    time - the mechanical enforcement of this project's own "every number
    traceable to config + seed + commit" standard."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}
    except Exception as e:  # noqa: BLE001 - provenance must never fail a run
        return {"commit": None, "branch": None, "dirty": None, "error": str(e)}


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - genuinely optional, absent is expected
        return None


def _gpu_driver_version() -> str | None:
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()
        )
        return out[0] if out else None
    except Exception:  # noqa: BLE001 - absent on non-NVIDIA/no-driver boxes (e.g. the Mac)
        return None


def capture_environment() -> dict:
    """Hardware/library versions - directly answers the real, repeated GPU
    driver-version/architecture incompatibilities this project has hit, which nothing previously
    recorded per-run."""
    gpu_available = torch.cuda.is_available()
    env = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "gpu_driver_version": _gpu_driver_version(),
        "cupy_version": _optional_version("cupy"),
        "crosssim_version": _optional_version("simulator"),
        "aihwkit_version": _optional_version("aihwkit"),
    }
    for var in _INSTANCE_ID_ENV_VARS:
        if var in os.environ:
            env["instance_id"] = os.environ[var]
            break
    return env

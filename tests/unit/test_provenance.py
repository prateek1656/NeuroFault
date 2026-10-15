"""neurofault.provenance is best-effort by design - every capture function
must degrade gracefully (never raise) when git/GPU/an optional backend is
absent, since a run must never fail just because provenance couldn't be
fully captured."""

from __future__ import annotations

import subprocess

import torch

from neurofault import provenance
from neurofault.provenance import capture_environment, capture_git_info


class TestCaptureGitInfo:
    def test_real_repo_returns_real_commit_and_branch(self):
        info = capture_git_info()
        assert info["commit"] is not None
        assert len(info["commit"]) == 40
        assert all(c in "0123456789abcdef" for c in info["commit"])
        assert info["branch"]
        assert isinstance(info["dirty"], bool)

    def test_git_unavailable_falls_back_gracefully(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _raise)
        info = capture_git_info()
        assert info == {
            "commit": None,
            "branch": None,
            "dirty": None,
            "error": "git not found",
        }

    def test_git_command_failure_falls_back_gracefully(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(subprocess, "run", _raise)
        info = capture_git_info()
        assert info["commit"] is None
        assert info["dirty"] is None
        assert "error" in info


class TestCaptureEnvironment:
    def test_returns_expected_keys(self):
        env = capture_environment()
        for key in (
            "platform",
            "python_version",
            "hostname",
            "torch_version",
            "torch_cuda_version",
            "gpu_available",
            "gpu_name",
            "gpu_driver_version",
            "cupy_version",
            "crosssim_version",
            "aihwkit_version",
        ):
            assert key in env

    def test_torch_version_matches_real_installed_torch(self):
        env = capture_environment()
        assert env["torch_version"] == torch.__version__

    def test_gpu_fields_consistent_with_cuda_availability(self):
        env = capture_environment()
        assert env["gpu_available"] == torch.cuda.is_available()
        if not torch.cuda.is_available():
            assert env["gpu_name"] is None

    def test_missing_optional_backend_reports_none_not_a_crash(self):
        env = capture_environment()
        # The development-time environment may or may not have crosssim/aihwkit/cupy
        # installed - either a real version string or None, never a raised
        # exception reaching the caller.
        assert env["crosssim_version"] is None or isinstance(env["crosssim_version"], str)
        assert env["aihwkit_version"] is None or isinstance(env["aihwkit_version"], str)
        assert env["cupy_version"] is None or isinstance(env["cupy_version"], str)

    def test_instance_id_is_captured_when_present(self, monkeypatch):
        for var in provenance._INSTANCE_ID_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CONTAINER_ID", "C.123456")
        env = capture_environment()
        assert env["instance_id"] == "C.123456"

    def test_instance_id_absent_by_default(self, monkeypatch):
        for var in provenance._INSTANCE_ID_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        env = capture_environment()
        assert "instance_id" not in env

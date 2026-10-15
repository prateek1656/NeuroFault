# Generic CUDA image for the full-scale CIFAR-10 baseline/mitigation runs; runs on any
# machine with an NVIDIA GPU and container runtime. Only needed once a CPU-only run's real,
# timed cost becomes prohibitive - verify on the actual machine
# (see CrossbarConfig.use_gpu's docstring) before trusting a long run here.
#
# CUDA 12.4 **devel** (not runtime) on Ubuntu 22.04, matching cupy-cuda12x
# (see pyproject.toml's "gpu" extra). Originally built on `runtime` on the
# (correct, at the time) assumption that no compilation happens in this
# image - wrong: cupy JIT-compiles kernels at runtime via NVRTC (e.g.
# CrossSim's set_matrix()/.astype() path, hit on the very first patched
# layer), which needs the CUDA toolkit's C++ headers (cuda_fp16.h etc.) -
# only present in `-devel` images, not `-runtime`. The failure seen with a
# `-runtime` base: `cupy.cuda.compiler.CompileException: cannot open source
# file "cuda_fp16.h"`.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# git: required to install CrossSim, pinned to a tag (see pyproject.toml),
# not a PyPI package. python3.12 deliberately NOT installed via apt here -
# verified directly during development: Ubuntu 22.04's default repos only carry
# python3.10 (no 3.12 package, deadsnakes PPA needed otherwise) - uv manages
# its own interpreter instead, avoiding that entirely.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/
RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY experiments ./experiments

# All three extras baked in - a live `uv sync` of exactly these three (torch/CUDA/cupy +
# CrossSim's own git submodule pull + AIHWKit) took ~90 minutes on a slow connection, on top
# of the base image pull. Baking them into the image at build time (once, on a fast
# connection) instead of installing live on every fresh machine is the single biggest lever
# found for cutting that setup time.
#
# UV_HTTP_TIMEOUT raised from uv's 30s default - real failure hit on this
# local build machine mid-download of one large wheel
# (nvidia-cudnn-cu13, ~700MB+) during a transient local network slowdown:
# "Failed to download distribution due to network timeout." A large-wheel
# download briefly dipping below uv's default per-request timeout is a real
# failure mode independent of which network is doing the downloading - 120s gives real headroom.
ENV UV_HTTP_TIMEOUT=120
RUN uv sync --frozen --extra crosssim --extra gpu --extra aihwkit

# Pre-download CIFAR-10 into ./data at build time - real, measured cost on a
# remote box: 170MB from torchvision's default mirror (University of
# Toronto) at ~65-100kB/s took 25-30+ minutes on two separate hosts with
# otherwise-good bandwidth, and outright HUNG (zero bytes transferred for
# minutes, confirmed via direct network measurement, not just "slow") on
# this local build machine too, even on a fast connection - the mirror
# itself is unreliable, not any specific host or network. Deliberately
# baked into the image rather than a persistent volume: CIFAR-10
# is 170MB and 100% static/read-only, ~2% of this image's own size -
# volumes are the right call for large or frequently-changing data (or
# writable state like checkpoints), but add real cost here (host-locking,
# recurring storage charges, another thing to manage) for a dataset
# this small and unchanging.
#
# torchvision's own downloader (urllib-based) has no timeout cap - it just
# hangs forever on a dead connection instead of failing and retrying,
# which is exactly what was observed. curl with explicit retry/timeout
# flags fetches the tarball directly into the exact path/filename
# torchvision expects (verified via torchvision.datasets.CIFAR10.url/
# .filename/.tgz_md5, not guessed); the download=True call below then
# finds it already present with a matching checksum and only extracts
# it locally - no network involved in that step at all.
# -C - (resume): a real retry happened on the first attempt at building this
# exact layer - curl restarted the whole 162MB download from byte 0 instead
# of resuming, because -C - wasn't set. Without it, a connection that drops
# periodically can burn through --retry's budget on repeated from-scratch
# attempts and never actually finish.
#
# --max-time is a ceiling on the WHOLE transfer, not a per-attempt/per-retry
# budget - the first version of this fix set it to 300s (5 minutes) while
# the real download, at this connection's measured ~65-70kB/s, genuinely
# needs ~40 minutes: `curl: (28) Operation timed out after 299097
# milliseconds with 20128000 out of 170498071 bytes received` - a
# self-inflicted failure, not a real network problem. 3600s gives real
# headroom over the ~40min expected duration plus retry overhead.
RUN mkdir -p ./data && \
    curl -fL -C - --retry 8 --retry-delay 10 --retry-all-errors \
      --connect-timeout 20 --max-time 3600 \
      -o ./data/cifar-10-python.tar.gz \
      https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && \
    uv run python -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True); torchvision.datasets.CIFAR10(root='./data', train=False, download=True)"

# KMP_DUPLICATE_LIB_OK: see experiments/run.py's own module docstring - the
# macOS-specific reason doesn't apply on Linux, but setting it is harmless
# and keeps this image's env consistent with the local dev entrypoint.
ENV KMP_DUPLICATE_LIB_OK=TRUE

ENTRYPOINT ["uv", "run", "experiments/run.py"]

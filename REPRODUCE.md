# Reproducing the papers

Three levels, from cheapest to most expensive.

| Level | What | Hardware | Needs |
|---|---|---|---|
| L1 | Re-derive every paper number from the shipped results | CPU | `results/` as shipped (no checkpoints, logs or raw archive) |
| L2 | Regenerate figures and build the papers | CPU | L1 outputs, `tectonic` |
| L3 | Re-run experiments | GPU | datasets, pre-deployment checkpoints, activation caches (Paper B) |

Setup for all levels:

```bash
uv sync --extra crosssim
```

All commands run from the repository root. The analysis scripts read `results/` and write to `results/derived/`. The figure scripts write to `papers/controller/figs/` and `papers/spectral/figs/`.

## What ships in `results/`

- Paper A runs (everything under `results/` except `results/spectral/` and `results/derived/`): per run copy, `events.json`, `config.json`, `metrics.json` and `run_metadata.json`. Logs, plots, validation reports and model checkpoints are not shipped. Some old runs lack `events.json` or `run_metadata.json`.
- Paper B runs (`results/spectral/<experiment>/<run>/`): per run `config.json`, `params.json`, `run_metadata.json`, `verdict.json` and the metric and summary CSV and JSON files. Raw caches (`_cells*`, `_cache`), activation caches and checkpoints are not shipped.
- `results/derived/`: the outputs of all analysis scripts, including the two audits that cannot be re-run here (see below).
- `results/MANIFEST.md` indexes the run families. `results/SUMMARY.md` is a per-group summary of the early runs.

## L1. Re-derive the paper numbers (CPU)

The scripts are deterministic. Run them in this order, because later ones read the CSVs written by earlier ones.

```bash
# 1. Canonical per-run table, per-claim values, learning-rate selection.
#    Reads results/**/events.json (not results/spectral).
#    Writes results/derived/canonical_runs.csv, claims.csv, canary_selection.csv
uv run python scripts/analysis/derive_paper_numbers.py

# 2. Paired statistics for every comparison in Paper A.
#    Reads canonical_runs.csv, dup_audit.csv, pair_audit.csv.
#    Writes results/derived/stats_final.csv and stats-final.md.
uv run python scripts/analysis/paper_stats.py

# 3. E1/E2/E3 analysis (the pre-registered replication and reviewer baselines).
#    Reads events.json directly and stats_final.csv.
#    Writes results/derived/e1e3_stats.csv and e1e3-results.md.
uv run python scripts/analysis/e1e3_stats.py

# 4. Lifetime, spares and gate-drift metrics used in Paper A.
#    Prints to stdout; writes nothing.
uv run python scripts/analysis/paper_a_metrics.py

# 5. Living spares at the tuned rate (the +0.03 control).
#    Reads the control's runs (results/_live20).
#    Writes results/derived/living-tuned-results.md.
uv run python scripts/analysis/living_tuned_stats.py
```

Audits (outputs shipped, cannot be re-run from this repository):

`scripts/analysis/audit_runs.py` (validity and provenance audit of every run directory) and `scripts/analysis/audit_classical.py` (audit of the fault-free baselines and classical static-mitigation runs) need the raw run archive: run logs, model checkpoints and per-run validation reports. That archive is not part of this repository. Their outputs are shipped in `results/derived/`: `pair_audit.csv`, `dup_audit.csv`, `predeploy_checkpoints.csv` and `classical_claims.csv`. `paper_stats.py` and `derive_paper_numbers.py` read the first two. The audit scripts are included so the method can be inspected.

Paper B verdicts are decided mechanically by `neurofault.spectral.verdict` against the frozen criteria in `experiments/spectral/criteria/`. Each run's `verdict.json` is shipped.

The `claims.csv` status column (`MATCH`, `ROUNDING`, `MISMATCH`) compares a re-derived value with the value the paper states. `MATCH` means within 0.05 points, `ROUNDING` within 0.10.

Summary table of the early run groups (prints to stdout):

```bash
uv run python scripts/summarize_results.py --root results
```

The per-run validation reports are not shipped, so its verdict column reads `UNKNOWN`. The shipped `results/SUMMARY.md` keeps the verdicts of the original generation, plus hand-written notes. Do not overwrite it with `--out`.

Caveats for L1:

- The statistical rules (pairing by seed, paired t test as primary, Holm families, verdict labels) are documented in the docstring of `scripts/analysis/paper_stats.py` and in Section 3 of Paper A.

## L2. Figures and papers (CPU)

```bash
# Paper A figures (fig1 is TikZ, built with tectonic; fig2-fig7 are matplotlib)
uv run python scripts/figures/paper_a/make_all.py

# Paper B figures. Reads only the shipped CSVs under results/spectral.
# fig_bidirectional is TikZ: scripts/figures/paper_b/fig_bidirectional.tex
uv run python scripts/figures/paper_b/make_figs.py

# Build the papers
tectonic papers/controller/main.tex
tectonic papers/spectral/main.tex
```

Install [tectonic](https://tectonic-typesetting.github.io/) separately. The Paper A LaTeX source shows a framed placeholder in place of any figure PDF that is missing, so the paper builds before figures are generated.

## L3. Re-run experiments (GPU)

Compute: the main Paper A results rest on a few hundred complete deployment lifetimes on ResNet-9 (the run audit checked 683), each needing a GPU. The repository does not record GPU-hours per experiment. See the papers for run counts and seeds.

### Data

CIFAR-10 and CIFAR-100 are downloaded by torchvision on first use into `./data` (relative to the working directory) via `download=True`. No manual step is needed. Run from the repository root so the data lands in `./data`.

### Pre-deployment checkpoints

Deployment-regime runs start from a fault-free, trained pre-deployment checkpoint (`pre_deploy.pt`). The checkpoints used for the papers (about 581 MB, 11 files) are published as a GitHub Release asset.

TODO(release): link

Pass a checkpoint with `--pre-deploy-checkpoint`.

Paper B re-runs also need trained checkpoints and activation caches. These are not shipped in the repository and will be provided as a release asset. The Paper B scripts load a ResNet-9 checkpoint by seed from `results/_live2/resnet9_predeploy_seed<seed>/pre_deploy.pt` (`neurofault.spectral.tiles.checkpoint_path`) and read cached activation samples from `results/spectral/activations/<seed>/`. Place the release files at those paths. Without them, the Paper B experiment scripts cannot run; the shipped CSVs and verdicts are enough for L1 and L2.

### Paper A: one arm

Each arm is one YAML config:

```bash
uv run python experiments/run.py --config experiments/configs/<arm>.yaml \
    --seed 42 --pre-deploy-checkpoint <path/to/pre_deploy.pt>
```

Options (from `experiments/run.py`): `--config` (required), `--epochs`, `--num-batches`, `--train-samples`, `--test-samples`, `--simulator {crosssim,aihwkit}`, `--dataset {cifar10,cifar100,mnist}`, `--seed`, `--pre-deploy-checkpoint`. When `--seed` is given, the run name becomes `<name>_seed<seed>` so a seed sweep writes distinct result directories. Results go to `results/<name>/`.

The `config.json` stored with a run shows placeholder burst rates. The rates actually used are in the run's `events.json` under `effective_params`. Read those, not `config.json`.

### Paper B: experiment scripts

Each script in `experiments/spectral/` runs one registered experiment against its frozen criteria in `experiments/spectral/criteria/*.yaml`. Several share a CLI:

```bash
uv run python experiments/spectral/b_c1b.py [--smoke] [--allow-dirty] [--jobs 3] [--out DIR]
```

`--smoke` runs a reduced grid that never counts toward a verdict. Not every script takes every flag; check `--help`. Examples of the main scripts: `a_bnfold.py`, `a_gate0.py`, `a_hardening_deploy.py`, `a_deploy.py`, `a_refresh.py`, `b_c1b.py`, `b_vs_prior.py`, `b_concurrent.py`, `e4_xabft.py`, `e5_weiroth_fair.py`, `f0_1_stuck_scaling.py`, `f0_8_bridge.py`. Their configs are in `experiments/spectral/configs/`. See [RESULTS.md](RESULTS.md) for which script produced which paper table.

Each run directory holds `config.json`, `params.json`, `run_metadata.json` (config hash, checkpoint SHA-256s, library versions) and `verdict.json`. The verdict records the SHA-256 of the criteria file used.

The criteria files are frozen. Their change policy is in `experiments/spectral/criteria/README.md`: a threshold may change only in a new commit, with the reason logged, and never after seeing the results it would judge.

### Seeds and fault maps

- Each seed trains one checkpoint. Within one seed and one GPU, all arms share the same checkpoint and the same fault maps, and the run records a fault-map hash per wave. The audit checks that these hashes are equal across arms.
- The same seed gives different fault maps on different GPU models. In the audit, all 9 multi-GPU groups differ. The pooled eight-seed main test mixes two GPU models (seeds 103 and 104 ran on a different GPU than 42 to 102). A re-run on other hardware will not reproduce the exact fault maps, and so will not reproduce per-seed numbers to the last digit.
- No stored run records its code commit (0 of 683). One snapshot commit held the code the runs used. The run audit documents this gap.
- Ten identical reruns differ by an RMS of 0.46 points on final accuracy. A single-seed gap below 1.22 points (the largest difference seen between identical reruns) is not interpretable.

### Tests

```bash
uv run pytest tests/ -q                    # full suite
uv run pytest tests/ -q -m "not slow"      # fast loop; logic tests use a tiny CNN
```

The `slow` marker selects tests that run a real ResNet-9 through CrossSim. The `statistical` marker selects many-trial goodness-of-fit checks with a small, documented false-failure rate.

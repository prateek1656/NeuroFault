# neurofault

A simulation study of fault-tolerant memristive crossbars for deep-neural-network inference. It asks two questions:

1. What repairs an aging crossbar? Cells fail while the chip is deployed, and a finite pool of spare cells is the repair budget.
2. How can a deployed chip find its faulty cells with few reads, without re-reading every cell?

The repository holds the simulator code (built on the CrossSim backend), the experiment configs and scripts, the shipped results, the analysis scripts that re-derive every number in the papers, and the two papers.

Everything here is simulation. There is no silicon and no device measurement.

## The two papers

### Paper A: "It Was the Learning Rate" (`papers/controller/`)

Full title: *It Was the Learning Rate: Compute-Matched Baselines Erase the Apparent Gains of a Memristor Crossbar Repair Controller.*

This is largely a negative and clarifying result. We built a repair controller (D) that paces spare use and gates retraining on a canary set. In a ResNet-9 / CIFAR-10 simulation (three fault waves, 15% spares), an early version looked far better than its rival. Most of that lead came from an untuned learning rate and unequal compute.

What the paper reports:

- Remap to a frozen reference can overwrite learned weights, and "living spares" avoid this by construction. A pre-registered tuned-rate control found no accuracy difference (+0.03 points, 3 seeds). An earlier +2.48 came from a barely working rate.
- A forward-only BatchNorm refit stays at chance once the spare pool is exhausted (76.06 points below the controller, 0 of 8 seeds).
- D beats equal-compute retraining-only by 3.78 points (5 of 5 seeds). The seeds were extended from 3 to 5 after a first look.
- D's apparent 34.0-point lead over a living-spares rival shrinks to 4.2 points once that rival's learning rate is tuned, to 1.06 once compute is matched, and to -0.005 in a pre-registered eight-seed replication.
- On final accuracy D is not distinguishable from the matched rival or from a one-shot Liu-style retrain. The paper ran no equivalence test, and results vary across seeds. On two of eight seeds D's canary gate drifted and skipped a needed burst.
- D has no edge or trails on calibrated drift and with many small waves.
- The paper also catalogues simulator errors we found and fixed (see `docs/simulator-bugs.md`).

The paper's conclusion is that a repair controller needs a tuned, compute-matched baseline and a replication before its margin means anything.

### Paper B: "Don't Read Every Cell" (`papers/spectral/`)

Full title: *Don't Read Every Cell: Few-Read Fault Localization and Drift Calibration in Analog Crossbars.*

The paper asks how many counted probe reads localize k stuck cells in a 256x256 tile. It uses a sparse-recovery decoder over random +-1 probes, with forward reads alone or jointly with transposed reads. It compares against reimplementations of published schemes under pre-registered criteria.

- At 1% read noise the joint decoder localizes up to 64 stuck cells with 4 to 12 reads. For k >= 4 it needs 4.3 to 6.4x fewer reads than a hybrid reimplementation of a multi-outlier code.
- Most of that factor comes from the sparse-recovery decoder, not from the second read direction. The benefit of the second direction is small or absent at 2% noise on tiles of 256 or more. Two registered claims about transposed reads were killed.
- Against a tuned X-ABFT baseline the decoder needs at least 2x fewer reads, or X-ABFT misses F1 0.9.
- A 16-parameter drift field fitted per tile from probe reads gives a digital correction that restores programmed accuracy (+2.1 to +13.3 points over the best cheap compensation on CIFAR-10 from 1 h to 1 yr). Applying the correction in the array recovers much less.
- The paper proves only a sufficient condition for exact recovery. It lists limits L1-L21 covering ideas that failed.
- The first-wave criteria were committed after nine full runs, so pre-registration was not clean for that wave. The paper says so.

## Quickstart

Requires Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prateek1656/NeuroFault
cd NeuroFault
uv sync --extra crosssim        # CrossSim v3.2.1, installed from git
uv run pytest -m "not slow"     # fast test loop
```

Notes:

- CrossSim is not on PyPI. The `crosssim` extra pins `git+https://github.com/sandialabs/cross-sim.git@v3.2.1`. Its import name is `simulator`.
- Optional extras: `aihwkit` (second backend, `aihwkit>=1.1.0`) and `gpu` (`cupy-cuda12x`, only for CUDA machines with `crossbar.use_gpu=true`).
- Tests marked `slow` run a real ResNet-9 through CrossSim. Tests marked `statistical` are many-trial goodness-of-fit checks with a small, documented false-failure rate.

To reproduce the paper numbers, figures and papers, see [REPRODUCE.md](REPRODUCE.md). For which script and result files produce each table and figure, see [RESULTS.md](RESULTS.md).

## Repository map

| Path | Contents |
|---|---|
| `src/neurofault/` | Library: crossbar simulation on CrossSim, faults, devices, mitigation (remap, spares), models, retraining controller, spectral detection and localization, closed loop |
| `experiments/run.py`, `experiments/configs/` | One experiment arm per YAML config (`python experiments/run.py --config <yaml>`) |
| `experiments/spectral/` | Paper B experiment scripts, `configs/`, and `criteria/` (frozen pre-registered pass/fail criteria) |
| `scripts/analysis/` | Scripts that re-derive every paper number from the shipped results |
| `scripts/figures/paper_a/`, `scripts/figures/paper_b/` | Figure scripts |
| `scripts/summarize_results.py` | Summary table of runs |
| `results/` | Trimmed run outputs: per run `events.json`, `config.json`, `metrics.json`, `run_metadata.json` for Paper A; per run config, params, verdict and metric CSV/JSON files under `results/spectral/<exp>/<run>/` for Paper B. Logs, plots, checkpoints and caches are not shipped. `MANIFEST.md` indexes the run families |
| `results/derived/` | Canonical CSVs: `canonical_runs`, `claims`, `stats_final`, `e1e3_stats`, `canary_selection`, `classical_claims`, `pair_audit`, `predeploy_checkpoints` |
| `papers/controller/`, `papers/spectral/` | LaTeX sources, `refs.bib` and figures for Papers A and B (build with tectonic) |
| `tests/` | pytest suite |
| `docs/simulator-bugs.md` | Catalogue of simulator errors found and fixed |

## Limits

- Simulation only. All results come from simulators with faults injected by us. Nothing is measured on silicon.
- Paper A: one simulator (CrossSim), stuck-at faults, ResNet-9 on CIFAR-10 for the main test. The main runs have no read noise, programming error or drift. The fault map is an oracle in the main runs. The closed-loop test at 2% read noise narrows this, and its 4% extension was killed.
- Paper A: D has no measured margin over matched rivals on final accuracy. The +3.78 over retraining-only is a five-seed, sequential-design result. The gate-drift diagnosis is post-hoc and rests on two seeds. Retraining compute and write endurance are not costed.
- Paper A provenance: no run records its code commit. The same seed gives different fault maps on different GPU models, and pooled seeds mix two GPU models.
- Paper B: stuck-at faults only, on an otherwise ideal tile, with the stored digital weights as reference. One architecture (ResNet-9), 256x256 tiles, PCM constants, three checkpoint seeds per dataset.
- Paper B: the digital-twin bridge to CrossSim did not meet its agreement bars. Absolute accuracies are not validated to 0.5 points, though relative comparisons are conservative.
- Paper B: a transposed read assumes bidirectional periphery whose cost is unverified. Read noise is white and independent across directions.

Each paper has a full limitations section. Read it before relying on any number.

## About

This work grew out of the author's 2025 M.Tech dissertation and was substantially rebuilt and extended independently in 2026.

Patent notice: some methods implemented here (in `src/neurofault/spectral/` and `src/neurofault/retraining/`) are the subject of pending patent applications. The BSD-3-Clause-Clear license grants no patent rights.

## How to cite

See [CITATION.cff](CITATION.cff). It lists the software and both papers (preprints).

## License

BSD-3-Clause-Clear. See [LICENSE](LICENSE).

# Results provenance

For each table and figure in the two papers: which script produces it, and which result files feed it. Where the mapping is not documented, the entry says "see script X". Commands to run the scripts are in [REPRODUCE.md](REPRODUCE.md).

Paths are relative to the repository root.

## Paper A: `papers/controller/`

### Data flow

1. Raw run outputs: `results/<run>/events.json` (plus `config.json`, `metrics.json`, `run_metadata.json`). Several run families sit one level down, in `results/_live*/<run>/`. Logs, plots and checkpoints are not shipped.
2. `scripts/analysis/derive_paper_numbers.py` collects the canonical run for each (run name, seed) (the newest complete copy wins when a run appears in several directories) and writes `results/derived/canonical_runs.csv`, `claims.csv` and `canary_selection.csv`. `claims.csv` has one row per reported number, with its backing runs.
3. `scripts/analysis/paper_stats.py` writes `results/derived/stats_final.csv` (paired statistics for every comparison).
4. `scripts/analysis/e1e3_stats.py` writes `results/derived/e1e3_stats.csv` (E1/E2/E3 tests, pre-registered).
5. `scripts/analysis/audit_runs.py` wrote `results/derived/pair_audit.csv` (fault-map pairing per comparison and seed), `dup_audit.csv` and `predeploy_checkpoints.csv`. `scripts/analysis/audit_classical.py` wrote `classical_claims.csv`. Both audits need the raw run archive (logs, checkpoints, validation reports), which is not shipped, so only their outputs are in the repository.

### Tables

| Label | Content | Produced by | Feeds from |
|---|---|---|---|
| `tab:positioning` | Closest prior work | Hand-written, no data | `papers/controller/refs.bib` |
| `tab:arms` | Arm definitions (R0, R1, B, B-matched, C, C', C'-matched, D, E1, E2 and others) | Hand-written. Arms are defined by the YAML files in `experiments/configs/` | `experiments/configs/` |
| `tab:ledger` | Methodology ledger (controls and their limits) | Hand-written. The audit figures come from `scripts/analysis/audit_runs.py` | `results/derived/pair_audit.csv`; see `audit_runs.py` for the other audit outputs |
| `tab:bugs` | Simulator pitfalls with measured effects | Hand-written. Catalogue in [docs/simulator-bugs.md](docs/simulator-bugs.md). Numbers: see `scripts/analysis/audit_classical.py` and `derive_paper_numbers.py` | `results/derived/classical_claims.csv`, `claims.csv` |
| `tab:main` | Main paired contrasts, 5 and 8 seeds | `scripts/analysis/paper_stats.py` (families H and S) and `scripts/analysis/e1e3_stats.py` (family E and the 8-seed D vs C'-matched) | `results/derived/stats_final.csv`, `e1e3_stats.csv`, `canonical_runs.csv` |
| `tab:lr` | Learning-rate selection (test-picked and canary-picked) | `scripts/analysis/derive_paper_numbers.py` | `results/derived/canary_selection.csv`; see also `scripts/analysis/paper_a_metrics.py` (canary pick rules) |
| `tab:tradeoffs` | Designs, final accuracy, lifetime and cost | Lifetime columns: `scripts/analysis/paper_a_metrics.py` (prints to stdout). Final accuracy and n: `derive_paper_numbers.py` | `results/derived/canonical_runs.csv`, `results/**/events.json` |
| `tab:other` | D vs C'-matched and B-matched on VGG6, ViT-tiny, CIFAR-100, many small waves, calibrated drift | `scripts/analysis/paper_stats.py` | `results/derived/stats_final.csv`, `canonical_runs.csv` |

### Figures

All Paper A figure scripts load runs through `scripts/figures/paper_a/_common.py`, which reuses the canonical-run collector in `scripts/analysis/paper_a_metrics.py` and `derive_paper_numbers.py` over `results/**/events.json`. `scripts/figures/paper_a/make_all.py` builds all of them into `papers/controller/figs/`.

| Label | File | Script |
|---|---|---|
| `fig:overview` | `fig1_overview.pdf` | `scripts/figures/paper_a/fig1_overview.tex` (TikZ diagram, no measured data) |
| `fig:main` | `fig3_main.pdf` | `scripts/figures/paper_a/fig3_main.py` (final accuracy per arm, per seed) |
| `fig:tuning` | `fig7_tuning.pdf` | `scripts/figures/paper_a/fig7_tuning.py` (learning-rate sweeps) |
| `fig:replication` | `fig6_replication.pdf` | `scripts/figures/paper_a/fig6_replication.py` (5-seed and 8-seed D vs C'-matched) |
| `fig:overwrite` | `fig2_overwrite.pdf` | `scripts/figures/paper_a/fig2_overwrite.py` (frozen vs living reference, old burst rate) |
| `fig:cliff` | `fig5_spare_cliff.pdf` | `scripts/figures/paper_a/fig5_spare_cliff.py` |
| `fig:lifetime` | `fig4_lifetime.pdf` | `scripts/figures/paper_a/fig4_lifetime.py` (per-wave accuracy through life) |

The paper's headline numbers outside the tables (for example +3.78, -0.005, +0.03, 76.06) are rows of `results/derived/claims.csv` and `stats_final.csv`. The +0.03 living-spares control is computed by `scripts/analysis/living_tuned_stats.py`.

## Paper B: `papers/spectral/`

Each run directory is `results/spectral/<experiment>/<timestamp>_<hash>_s0/`. It holds `config.json`, `params.json`, `run_metadata.json`, `verdict.json` and metric CSVs. The experiment scripts are in `experiments/spectral/`, the criteria in `experiments/spectral/criteria/`.

Short names used in the paper source comments:

| Short | Experiment dir | Run | Script |
|---|---|---|---|
| BF | `a_bnfold_v2` | `20261010T022159_fca882d8_s0` | `experiments/spectral/a_bnfold.py` |
| G0 | `a_gate0` | `20261009T020655_ae67b976_s0` | `experiments/spectral/a_gate0.py` |
| E2 | `a_exp2` | `20261009T052819_52891c48_s0` | `experiments/spectral/a_exp2_adabs.py` (jobs: `a_exp2_jobs.py`) |
| F01 | `f0_1` | `20261009T023921_a8f8370e_s0` | `experiments/spectral/f0_1_stuck_scaling.py` |
| F03 | `f0_3` | `20261009T181003_aca2558e_s0` | `experiments/spectral/f0_3_headroom.py` |
| C1 | `b_c1_prereg` | `20261009T035140_0fdb78ed_s0` | `experiments/spectral/b_c1_prereg.py` |
| C1B | `b_c1b` | `20261009T115059_609a698a_s0` | `experiments/spectral/b_c1b.py` |
| E1C | `b_e1c` | `20261008T205034_9b7dee7d_s0` | `experiments/spectral/b_e0_e1c.py` |
| E1 | `b_e1` | `20261009T033411_3e0a818f_s0` | `experiments/spectral/b_e1.py` |
| CX1 | `c_x1` | `20261010T023044_1e4807e2_s0` | `experiments/spectral/c_x1.py` |
| H1 | `f0_8` | `20261011T151445_d931ec73_s0` | `experiments/spectral/f0_8_bridge.py` (diagnosis: `f0_8_diag.py`) |
| H2 | `a_hardening` | `20261010T220159_8434d92b_s0` | `experiments/spectral/a_hardening.py` |
| H2B | `a_hardening_deploy` | `20261011T041351_794c47e8_s0` | `experiments/spectral/a_hardening_deploy.py` |
| H5 | `a_deploy` | `20261010T223102_71522b22_s0` | `experiments/spectral/a_deploy.py` |
| H6 | `a_refresh` | `20261011T185337_449adbef_s0` | `experiments/spectral/a_refresh.py` |
| H7 | `b_concurrent` | `20261011T152741_d4b34e56_s0` | `experiments/spectral/b_concurrent.py` |
| H8 | `b_vs_prior` | `20261011T210412_0ec5ea78_s0` | `experiments/spectral/b_vs_prior.py` |

Verdicts are decided mechanically by `neurofault.spectral.verdict` against the frozen criteria. Each run's `verdict.json` is shipped.

### Tables

| Label | Content | Feeds from |
|---|---|---|
| `tab:position` | Positioning against prior art | Hand-written, no data |
| `tab:params` | Provenance of simulator constants | `src/neurofault/spectral/params.py`; each run dir's `params.json` |
| `tab:ledger` | Pre-registration ledger, mechanical outcomes | `verdict.json` in the run directory of each listed claim (G0, F0.0 `f0_0`, F01, F03, C1, C1B, E2, BF, H2B, H5, H6, H7, H8) |
| `tab:loc` | H8, E4, E5: mean reads, tile 256 | H8 run (`b_vs_prior`): `verdict_metrics.csv`, `eval_instances.csv`, and `results/spectral/b_vs_prior/selected_configs_full.json`; scripts `b_vs_prior.py`, `e4_xabft.py`, `e5_weiroth_fair.py` |
| `tab:c1b` | C1b reads ratio | C1B run: `verdict_metrics.csv` |
| `tab:tradeoff` | Localizers side by side | Reads from `tab:loc` (H8 run); other columns are text |
| `tab:h7` | Concurrent detection vs live inferences | H7 run (`b_concurrent`) |
| `tab:exp2` | M4 vs AdaBS, accuracy and cost | E2 run (`a_exp2`) |
| `tab:bf` | `a_bnfold` final accuracy, gain and CI | BF run (`a_bnfold_v2`) |
| `tab:bfverdict` | Verdicts under three budget readings | BF run: `verdict.json` |
| `tab:scen` | Strict-variant gain per scenario | BF run |
| `tab:h2` | Hardening: deployable M4 vs stronger baselines | H2B run: `summary.csv`; H2 run for the ideal-M4 numbers; H5 (`a_deploy`) for the deployable cost. See `a_hardening_deploy.py` and `a_deploy.py` |
| `tab:h6` | Refresh allocation, writes per cell per year | H6 run (`a_refresh`): `verdict.json` and final table: see `a_refresh.py` |
| `tab:limits` | Limits L1-L21 | `verdict.json` in the run directories cited in each row (F01, G0, F03, C1, E2, C1B, CX1, H5, H2B, H6, H7); rows marked tier A or S are analysis or throwaway probes, not registered runs |

### Figures

`scripts/figures/paper_b/make_figs.py` reads only CSVs under `results/spectral/` and writes to `papers/spectral/figs/`.

| Label | File | Source |
|---|---|---|
| `fig:bidir` | `fig_bidirectional.pdf` | `scripts/figures/paper_b/fig_bidirectional.tex` (TikZ diagram, no measured data) |
| `fig:ratio` | `fig_reads_ratio.pdf` | `make_figs.py` from C1B `verdict_metrics.csv` |
| `fig:gate0` | `fig_gate0.pdf` | `make_figs.py` from G0 `incumbent_vs_m4.csv` (model D3, sigma 0.02) |
| `fig:exp2` | `fig_exp2_pareto.pdf` | `make_figs.py` from E2 `pareto_W1.csv`, `pareto_W2.csv` |
| `fig:drift` | `fig_drift_accuracy.pdf` | `make_figs.py` from BF `summary_strict.csv`, `summary.csv` |
| `fig:s1` | `fig_sigma1_auc.pdf` | `make_figs.py` from F01 `tile_metrics.csv` (headline stage, `stuck_random`, per-cell noise 0.02, signed encoding) |

Not covered by the five source runs that `make_figs.py` loads: the closed-loop runs under `results/spectral/x_loop*`, which feed Paper A (closed-loop section) through `scripts/analysis/paper_a_metrics.py`. Their scripts are `experiments/spectral/x_loop.py`, `x_loop_summary.py` and `x_loop_ext_summary.py`.

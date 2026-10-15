# Results manifest

Hand-written index of the run families under `results/`. It says what each family is, whether the papers use it, and which families were superseded or produced under a since-fixed simulator bug (see [../docs/simulator-bugs.md](../docs/simulator-bugs.md)). The aggregated numbers of the early runs are in `SUMMARY.md`. The numbers of the papers are re-derived into `derived/` (see [../REPRODUCE.md](../REPRODUCE.md)).

What ships: per run copy `events.json`, `config.json`, `metrics.json` and `run_metadata.json`. Logs, plots, validation reports, checkpoints and per-run caches are not shipped. Paper B runs are under `spectral/` (see the last section).

Status labels:

- **Used**: feeds a number in a paper.
- **Record**: valid as a record of what the code did at the time, not a finding.
- **Superseded**: produced under a since-fixed bug or an old setup. Not used in the papers.
- **Smoke / probe**: small dry run or bug-isolation probe. Not a result.

## Early runs (small CNN and first ResNet-9 work)

| Result dir(s) | Status | What it is |
|---|---|---|
| `resnet9_ibm_recipe_hwa_full_calibrated_gpu_seed{202609241730,101,102,103}` | **Used** | Fault-free ResNet-9 baseline with the ADC/DAC calibration fixes: final 89.79% (population stdev 0.107 pp, sample stdev 0.124 pp) across 4 seeds. `SUMMARY.md` reports the sample stdev. |
| `no_mitigation_moderate_seed{42,7,123}` | Record | No-mitigation arm of the first mitigation comparison: moderate fault regime, small CNN, 1,000 test images. Seeds are 42, 7 and 123. The unseeded `no_mitigation_moderate` is an exact copy of seed 42, not a fourth seed. 3-seed mean 41.57 (39.6 / 41.7 / 43.4). Ran with the DAC-clipping and ADC-range defects active, so it is not an absolute-accuracy claim for a correct simulator. |
| `soft_mitigation_only_moderate_seed*` | Record | 3-seed mean 44.13 (42.2 / 45.3 / 44.9). The +2.57 over no mitigation is confounded twice: 20 versus 15 epochs, and a pre-fix dispatch that touched 25.9 to 26.1 million non-stuck cells per run. It is not evidence that soft mitigation works or does not. |
| `remapping_only_moderate_seed*` | Superseded | 3-seed mean 41.90 (41.2 / 40.1 / 44.4), a wash against no mitigation. Ran with both remap bugs and the converter defects, so it says nothing about remapping's capacity ceiling. |
| `layer_reset_only_moderate_seed*` | Superseded | 19.80%, 21.8 points below no mitigation. Produced under the pre-fix dispatch (reset touched 47 to 55 percent of one layer) and the converter defects. On ResNet-9 after the fixes layer reset is level with no mitigation (42.56 vs 42.64, n=1). |
| `combined_all_strategies_moderate_seed*` | Superseded | 37.93%. Soft mitigation was active (about 9.6 million touches), the arm ran 10 epochs against 15 for no mitigation, and both remap bugs and the converter defects were present. |
| `device_fefet_*_moderate_seed*` | Record | FeFET device replicate of the same arms. Converter defects and pre-fix dispatch active. Soft versus no mitigation is not resolved (3 seeds). |
| `device_sttmram_*_moderate*` | Not used in the papers | Numerically identical to the default device on CrossSim (no per-technology dynamics). No device comparison exists. |
| `no_mitigation_aihwkit_moderate_seed42`, `no_mitigation_mnist_moderate_seed42` | Smoke / probe | Only the no-mitigation arm was run on the second backend and on MNIST. |
| `baseline_no_fault_scaled_check_resnet_seed42`, `baseline_no_fault_scaled_check_deep_seed42`, `baseline_no_fault_scaled_check*` | Superseded | Pre-ResNet-9 architecture search. Produced through the converter defects, so they do not show that capacity was the bottleneck. |
| `baseline_no_fault_full_scale_resnet_gpu_seed20260922`, `resnet9_ibm_recipe_full_scale_gpu_seed*` | Superseded | SGD with one-cycle ResNet-9, before the ADC/DAC fixes. Plateaued at 27.12% / 30.18%. A documented negative result that led to the converter fixes. |
| `resnet9_ibm_recipe_hwa_calibrated_gpu_seed202609241000` | Superseded | Intermediate 45.98% after only the float-pretrain and ADC-calibration fixes, before the DAC fix (45.98 to 89.79 is unpaired: one seed against four). Its `max` of 88.90 is the float-phase peak, not an analog number. |
| `resnet9_ibm_recipe_hwa_pretrain_gpu_seed20260923*` | Smoke / probe | Isolation test of optimizer versus architecture. |
| `no_mitigation*`, `no_mitigation_crosssim*`, `no_mitigation_mnist*`, `baseline_no_fault`, `fault_aware_retraining`, `layer_reset_only`, `remapping_only`, `soft_mitigation_only`, `combined_all_strategies` (unseeded or original-regime) | Superseded | Early single-run baselines. The original regime (density 0.1 every 8 batches) saturates to 100 percent stuck and every arm ends at chance; the moderate regime above replaced it. |
| `smoke_*`, `*_smoke*`, `local_check_*`, `combined_smoke*`, `calibrate_adc_smoke*`, `pretrain_smoke*`, `resnet9_crossbar_probe_cpu_seed0` | Smoke / probe | Small dry runs and bug-isolation probes, never meant as results. `local_check_no_dac` and `local_check_with_dac` (24.80 / 37.40, seed 7, 500 test images, SE about 2.2 points) are a single-seed A/B of the DAC defect. |
| `calibration_sweep*` | Smoke / probe | Stub directories with no metrics. |

## Remap bugs: every remapping and combined result before the fix is superseded

`resnet9_fault_remapping_seed42`, `resnet9_fault_combined_seed42`, and every older remapping or combined result (`remapping_only*`, `combined_all_strategies*`, `device_*_remapping_*`, `device_*_combined_*`) ran under two bugs:

- **Selection stall**: remapped cells stayed in the stuck mask, so each later wave's candidate list began with already-remapped cells and selection stopped at the first one. Run logs show wave 1 remapping 14,745 of 14,745 cells, wave 2 remapping 0 of 14,745. About 5 percent of every pool was never used.
- **Health gate**: remap fired only when a per-layer average health exceeded 75. That health is a synthetic stress value that drifts from 100 to 90 with zero faults. Per-layer values were 85-86, 75.1-76.3 and 67.7-68.6 at waves 1, 2 and 3, so wave 2 passed by luck and wave 3 could never remap.

The runs remain valid as a record of what that code did. They do not measure remapping's capacity ceiling. Any `redundancy_factor` sweep run before the fix is invalid (rates 0.20 and 0.30 behaved like 0.15).

## Deployment-regime and later run families

The main Paper A runs. "Live dir" is the `results/_live*` directory that holds the family. Common to every deployment family:

- No run records its code commit.
- Fault maps are GPU-specific: the same seed on a different GPU model gives a different map. Maps are equal across arms inside one seed and one GPU.
- `config.json` shows placeholder burst rates. The rates actually used are in `events.json` under `effective_params`.
- The test set is the 5,000-image CIFAR-10 subset throughout. Seed counts are 3, or 5 for the main test, VGG6, ViT-tiny and the D2 main test.

| Result dir(s) (live dir) | Status | Finding and caveat |
|---|---|---|
| `resnet9_deploy_{r0,r1,a,b,b_matched,c,c_living,d,d_*,s2_*}` Stage 1 and 2 (`_live2`, `_live3`) | Record | Deployment regime, oracle fault detection, random stuck-at. Arms are paired (same checkpoint and maps). The burst rate 0.005 was under-tuned (grid edge), so the D 83.3 vs C' 49.3 vs B 38.6 gap is the "before tuning" baseline only. Ablations predate the matched-step re-runs. |
| `resnet9_deploy_*_eta*` burst-rate sweeps (`_live3`) | **Used** (single-seed points flagged) | Paired. Many rates are one seed. The tuned ResNet-9 rates were picked on seed-42 test accuracy (B 0.5, C' 0.5, D 0.3). A canary-based re-pick would cost at most 0.45 point (3 seeds). The stored config is authoritative. |
| `resnet9_deploy_*_stress*`, `*_stressM*` (`_live3`, 162 runs) | **Used** | Six stress cases (2, 5, 10 percent spares; 15, 20, 25 percent faults per wave). D vs C'-matched +0.49 (noise-level), +0.70 (mixed), +2.33, +2.69, +1.35, +2.14 (consistent). None significant after Holm correction. |
| `resnet9_deploy_*_g1..g6`, `gfix`, `matchDgfix` (`_live5`, `_live6`) | Record | Gate and priors variants. Single-seed rows are noisy (one moved 7.7 points with the priors file; a same-config repeat moved 0.46). Quote only the 3-seed `_live6` rows. |
| `resnet9_deploy_d_ew2`, `_ew6` (`_live3`, `_live5`, `_live6`) | Record | Name collision across dirs (old low-rate priors 55-76 versus tuned priors 87-88). Always qualify by dir. |
| `resnet9_deploy_{r1,c_living,d}_clu_seed42` (`_live5`) | Superseded | Clustered faults, n=1. Started before the clustered-selection fix, so they may have used the biased selector. Not used. |
| `resnet9_deploy_*_c100*` (`_live5`, 18 runs) | Superseded | First CIFAR-100 attempt. Every arm about 1 percent (weak base network, unsuitable burst rate). Replaced by `resnet9_next_deploy_*` (`_live6`) and `resnet9_final_deploy_*c100*` (`_live12`). |
| `resnet9_next_deploy_*` CIFAR-100 (`_live6`), `resnet9_final_deploy_*c100*` (`_live12`) | **Used** (rate caveat) | 3 seeds, paired, equal steps. D 63.3 vs best C'-matched 56.1 (+7.16, consistent; Holm 0.024 at 2 df, indicative). Rates were picked on seed-42 test accuracy; no canary alternative exists. |
| `resnet9_final_*` main test, 5 seeds (`_live11`; seeds 103 and 104 on a second GPU model) | **Used** | Paired within dir. D vs B-matched +3.78 (significant). D vs C'-matched +1.06, 5 of 5, unadjusted p=0.005, Holm 0.056 (consistent, not significant after correction). Seeds 103 and 104 were added after the first look, so the sign-test p=0.031 is not quoted as significance. |
| `vgg6_net2_*` (`_live6`, `_live7`, `_live12`) and `vgg6_final_*` | **Used** (caveat) | Paired within seed and dir. Canary-picked rate 0.3 (the only networks where the canary pick applies). D vs C'-matched -0.52 (3 of 5, mixed). |
| `vit_net3_*` (`_live8`, `_live10`) and `vit_final_*` | **Used** (caveat) | As VGG6, plus seeds 103 and 104 ran on a different GPU model than 42, 101 and 102, so pooled 5-seed numbers mix two fault-map generators. D vs C'-matched +1.50 (4 of 5, mixed; unadjusted p=0.038, Holm 0.29). |
| `resnet9_real_s1..s3_*` (`_live9`) | **Used** (caveat) | Realistic fault models: Weibull wear-out, stuck-at plus drift, row/column lines. Paired, steps match. D vs C'-matched +0.94 / +1.11 / +0.73, 3 of 3 each (S2 Holm 0.019 at 2 df, indicative). |
| `resnet9_calib_s4{a,b,a5,b5}_{r1,b_matched,d}` (`_live13`, `_live14`) | **Used** (caveat) | Hardware-calibrated faults (S4). D vs B-matched -0.65, -1.05 (mixed), -1.99, -3.54 (negative): D loses on average in all 4. D vs C'-matched in S4b5 is -4.95 (0 of 3; Holm 0.046); C'-matched ran only for S4b5. |
| `resnet9_calib_s4*_{d2,c2,b_matched,r1}` (`_live15`, `_live16`) | Record | D2/C2 vs B-matched are paired and step-matched. D2 minus D is unpaired and exploratory (different GPU models; fault-map hash differs at all 5 waves in 12 of 12 pairs; different step totals). Not a paired difference. |
| `resnet9_d2_*` D2 main test, 5 seeds (`_live17`) | Record | D2 vs C'-matched +0.19 (3 of 5, mixed). D2 vs B-matched +3.31 and D2 vs C2 +2.52 (significant). D2 minus D (-1.53, 0 of 5) has mixed provenance and is exploratory. |
| `resnet9_cont_c_living_seed42`, `resnet9_cont_d_seed42` (`_live4`) | Superseded | Continuous bridge, run before the gate-reference fix. D final 63.82 reflects that bug. Not used. |
| `resnet9_cont_next_*`, `resnet9_cont_d_fixgate` (`_live5`, `_live6`) | Record | Continuous regime. D vs C'-matched +1.29 (+1.36, +0.56, +1.96; 3 of 3; unadjusted p=0.086, Holm 0.43; n=3). The arms do not share a checkpoint (each trains its own network; start canary differs 0.1 to 0.7 point), so pairing is by seed only. Classic arms (`cont_next_classic_*`) are n=1 and unpaired (global-RNG fault maps against the seeded generator of C'/D). Stuck-weight freeze is off on this path. |
| `resnet9_fault_{no_mitigation,soft_mitigation,layer_reset,remapping,combined}_seed42` | Superseded | Early ResNet-9 fault runs with both remap bugs active. Remap used 656,868 of 985,293 spares. n=1. |
| `resnet9_fault_mixed_*`, `cont_next_classic_*` (seed 42) | Record | After the remap fix, n=1, freeze off, global-RNG fault maps. Classic arms at 15 percent spares are indistinguishable from each other and from no mitigation (run-to-run noise at n=1 about 1 point). The spare-fraction sweep 39.8 / 40.0 / 39.9 / 43.0 / 89.1 for 5, 10, 15, 20, 30 percent is a capacity cliff: 30 percent exceeds the 27.1 percent cumulative stuck fraction (1 - 0.9^3), so every stuck cell has a spare. Nothing between 20 and 30 percent was run. |
| R1 remap at 15% spares, deployment regime (all networks) | **Used** | At chance (9.7 percent on ResNet-9, 5 seeds; R1 minus R0 -0.38, p=0.66, n=3). At 30 percent spares it equals the clean chip. A capacity cliff, not a mitigation curve. |
| `resnet9_tuned_c_{frozen,living}_seed*` (`_live20`) | **Used** | Living-spares control at the tuned burst rate: +0.03 points, 3 seeds. See `derived/living-tuned-results.md`. |

| `resnet9_e1_liu_{first,last}`, `resnet9_e2_bn_recal_only` (`_live18`, `_live19`) | **Used** | Reviewer-baseline runs of the pre-registered replication: E1 one-shot Liu-style retrain, E2 forward-only BatchNorm refit (ends at chance once spares are exhausted). Seeds are split across the two dirs. |
| `resnet9_final_{d,c_living_matched,r1}` extra seeds (`_live18`) | **Used** | The extra seeds that extend the main test to 8 seeds for D vs C'-matched (-0.005). |

The closed-loop runs are under `spectral/x_loop*` (Paper A, closed-loop section). Which directory feeds which number is in `derived/canonical_runs.csv` (`source_dir`) and [../RESULTS.md](../RESULTS.md).

## Paper B runs (`spectral/`)

One directory per experiment, one run directory per execution (`spectral/<experiment>/<timestamp>_<hash>_s0/`). Each holds `config.json`, `params.json`, `run_metadata.json`, `verdict.json` and metric or summary CSV and JSON files. Raw caches (`_cells*`, `_cache`), activation caches and checkpoints are not shipped. [../RESULTS.md](../RESULTS.md) lists the run each paper table uses.

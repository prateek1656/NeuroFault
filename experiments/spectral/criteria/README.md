# Pre-registered criteria (spectral-diagnosis vertical)

These YAML files transcribe the pass/kill thresholds of the direction plans
(the author's planning notes, not published) so that `neurofault.spectral.verdict` can decide mechanically.

**Frozen as of the commit that adds them (frozen: 2026-10-08).** They must be committed before the
first full run of any arm they cover. Smoke runs never count.

Change policy: a threshold, comparator, aggregation, filter or metric name may only change in a new
commit, and every change is logged below with date, file, criterion id, old -> new and the reason.
The previous verdict is kept (verdict.json records the sha256 of the criteria file used). Never edit a
file after seeing results it would judge.

Publication note: the YAML files ship byte-identical to the versions the verdicts were scored
against (`tests/invariants/test_preregistration_hashes.py` checks every shipped `verdict.json`).
Their `source` / `source_plan` fields name sections of the author's planning notes ("master plan",
"plan-B", `docs/research/...`), which are not part of this repository; "provisional" there means
preliminary numbers.

Rules of transcription: numbers are copied from the plan as written. Where the plan is ambiguous
(quantifier, aggregation, unit) the criterion carries a `note:` saying how it was read; the note is part
of the freeze. Metric names are the contract with the experiment scripts (derived columns such as
ratios, slopes and per-row flags are produced by the script and documented in the `note`).
Schema and semantics: see the docstring at the top of `src/neurofault/spectral/verdict.py`.

| File | Plan source |
|---|---|
| f0_validity.yaml | plan-F0 §9 F0.0a-e, F0.8a-c |
| f0_stuck_scaling.yaml | plan-F0 §9 F0.1 (H1.1-H1.6) |
| f0_lowrank.yaml | plan-F0 §9 F0.2 (K2.1-K2.5) |
| f0_headroom.yaml | plan-F0 §9 F0.3 (H3.1-H3.5), F0.4 (K4.1) |
| f0_loop.yaml | plan-F0 §7.2 V1-V8, §9 F0.6 (K6.x), F0.7 (K7.x) |
| a_gate0.yaml | plan-A §4 Gate 0' (G0-a..f and kill/redirect) |
| a_success.yaml | plan-A §6 (S1-S4, K1-K5) |
| b_claims.yaml | plan-B §5 (C1-C6; C4 deferred per master plan v3; go/no-go) |
| c_claims.yaml | plan-C §2, §5, §8 (S1-S4, K1-K4, D1/D2, KX) |
| a_refresh.yaml | master plan §10 H6 (registered after provisional H5, before any H6 run) |
| a_hardening_deploy.yaml | master plan §10 H2b (registered after provisional H2/H5, before any H2b run) |
| b_concurrent.yaml | master plan §10 H7 = plan-B E5 / C5 (registered 2026-10-11 after smoke + single-tile design probes, before any full run) |
| b_vs_prior.yaml | master plan §10 H8: registered confirmation of CS localizer vs HPE / Wei & Roth (2026-10-11, after exploratory b_e1, before any full run) |
| x_loop.yaml | master plan X.1/X.2 closed loop: item-4 controller D on a probe-estimated fault map vs the oracle map, dense (S1) + sparse regimes (2026-10-12, before any full run) |
| x_loop_ext.yaml | closed-loop extension: ResNet-9 seeds 103/104, read noise 4%, ViT-tiny (2026-10-12, before any counted run) |

## Change log

- 2026-10-09, `a_gate0.yaml` (G0-d, GK-dead, GK-crossover-only filters on `t_s`): `1.0e4` / `1.0e8`
  rewritten as `10000.0` / `100000000.0`. Reason: PyYAML (YAML 1.1) parses `1.0e4` as a STRING, so
  the filter compared against text and the clauses silently evaluated as "no data". Same numbers,
  representation fix only - no threshold changed. Found by the A Gate 0' build agent; fixed by the
  main session. All other criteria files scanned: no other numeric value parses as a string.
- 2026-10-09, `f0_loop.yaml` V6 -> V6p. Reason: V6 was transcribed with the wrong source claim. The
  "~10% error at 10% variation" in Sun et al. PNAS 2019 is about the linear-system (inversion) circuit's
  output voltages, not eigenvectors (verified by full-text read);
  research-01 had the qualifier, plan-F0 V6 dropped it. V6 is kept as a non-scoring record; V6p checks the
  simulator against first-order perturbation theory (currently 0.0113 vs 0.0113) plus 1/gap scaling. The
  1/gap slope band [-1.3, -0.7] is set by the main session, not by a plan - flagged as such. Unblocks C'.

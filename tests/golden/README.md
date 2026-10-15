# Remap golden outputs

Deterministic CPU run of `remap_golden_config.yaml` (CrossSim, ResNet-9, 24 batches, density 0.1,
waves at batches 0/6/12/18, health_check_interval 2, remapping on, seed 7). Two runs were
bit-identical (metrics.json minus `epoch_seconds`, and layer_health.json).

Generated from the code before the remap capacity / re-failure changes.

Command (set `output_dir:` in a copy of the config; run.py has no --output-dir flag):

    uv run experiments/run.py --config <copy of tests/golden/remap_golden_config.yaml>

Files: `remap_golden_metrics.json` (epoch_seconds stripped), `remap_golden_layer_health.json`.

This reflects PRE-bug-fix behavior: remap bug #1 (selection starts at already-remapped cells, so
later "Remapped 0/N" with capacity left) and bug #2 (per-layer avg_health <= 75 gate: waves at
batch 12 and 18 never attempt remap). A regression test after the fixes must expect differences
only where remap fires after wave 1 (remappings counter, health/stuck effects downstream);
loss/accuracy/faults must stay byte-equal through batch 6 (the first post-wave-1 remap
happens in the forward pass of batch 7).

# Refactor golden outputs

Deterministic CPU run of `refactor_golden_config.yaml` (CrossSim, ResNet-9, 6 epochs x 4 batches,
epoch 0 = float pretrain, waves at global batches 6/12/18, density 0.1 primary stuck-at plus a
drift secondary_fault, health_check_interval 2, seed 7). Remapping is DISABLED, soft_mitigation
and layer_reset are enabled, so remap/dispatch bug fixes  cannot change it: it guards the
run.py training-loop refactor . Soft mitigations and layer resets both fire (9 resets).
Two runs were bit-identical (metrics.json minus `epoch_seconds`, and layer_health.json).

Generated from the code immediately before the run.py training-loop refactor, via a clean
`git worktree`, run with PYTHONPATH pointing at the worktree's src/.

Files: `refactor_golden_metrics.json` (epoch_seconds stripped), `refactor_golden_layer_health.json`.
Regression: tests/invariants/test_refactor_golden.py asserts byte-equality of the continuous path.

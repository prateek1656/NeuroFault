import torch

from neurofault.crossbar.self_healing import RedundancyPool, RowColumnRedundancyPool
from neurofault.mitigation.remap import apply_remap
from tests.unit.fakes import make_handle


def test_remap_writes_through_the_same_canonical_accessor():
    """Direct regression test for bug #1: the original codebase built a
    second, disconnected Crossbar inside SelfHealingCrossbar and remapping
    never affected the object the model actually computed with."""
    handle = make_handle(shape=(4, 4))
    pool = RedundancyPool(handle, redundancy_factor=0.5)

    reference_value = pool.reference_conductance[0, 0].item()
    # Corrupt the cell after the reference was captured, simulating a fault,
    # via the same accessor apply_remap will read/write.
    matrix = handle.accessor.read()
    matrix[0, 0] = 999.0
    handle.accessor.write(matrix)

    remapped = apply_remap(handle, pool, [(0, 0)])

    assert remapped == 1
    # The write must land on the SAME handle's accessor apply_remap was given.
    assert handle.accessor.read()[0, 0].item() == reference_value

    # A second, independently-constructed handle must be untouched - this is
    # what "disconnected shadow crossbar" looked like in the original bug.
    other_handle = make_handle(shape=(4, 4))
    other_matrix = other_handle.accessor.read()
    other_matrix[0, 0] = 999.0
    other_handle.accessor.write(other_matrix)
    assert other_handle.accessor.read()[0, 0].item() == 999.0


def test_remap_stops_when_redundancy_capacity_exhausted():
    handle = make_handle(shape=(4, 4))
    pool = RedundancyPool(handle, redundancy_factor=0.1)  # capacity = max(1, int(16*0.1)) = 1

    remapped = apply_remap(handle, pool, [(0, 0), (1, 1), (2, 2)])

    assert remapped == 1
    assert pool.available_capacity == 0


def test_remap_stops_the_whole_batch_at_the_first_already_remapped_position():
    """A repeat mid-list must stop the batch at that point, not just skip the
    repeat and continue - matching remap_device()'s original per-call
    early-break semantics (see self_healing.py's RedundancyPool.remap_device
    docstring). This is the one subtle behavior the 2026-09-29 vectorization
    (RedundancyPool.select_remappable) had to preserve exactly, since a naive
    rewrite could easily turn "stop" into "skip and continue"."""
    handle = make_handle(shape=(4, 4))
    pool = RedundancyPool(handle, redundancy_factor=1.0)  # capacity = 16, never the limiter here

    first = apply_remap(handle, pool, [(0, 0)])
    assert first == 1

    # (0, 0) is already remapped; it sits in the middle of this batch.
    second = apply_remap(handle, pool, [(1, 1), (0, 0), (2, 2)])

    assert second == 1  # only (1, 1) - stopped at the repeat, (2, 2) never touched
    assert pool.available_capacity == 16 - 2  # (0, 0) and (1, 1) only


def test_apply_remap_dispatches_to_row_col_granularity_via_pool_type():
    """apply_remap() must branch on pool type, not require dispatch.py (or
    any other caller) to know which redundancy scheme is active."""
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.5, condemn_threshold=0.5)

    reference_row = pool.reference_conductance[0, :].clone()
    matrix = handle.accessor.read()
    matrix[0, :] = 999.0  # corrupt every cell in row 0
    handle.accessor.write(matrix)

    critical = [(0, c) for c in range(10)]  # whole row critical -> condemns row 0
    remapped = apply_remap(handle, pool, critical)

    assert remapped == 1  # one row repaired, per the row/col-granularity return convention
    assert torch.equal(handle.accessor.read()[0, :], reference_row)


def test_apply_remap_row_col_granularity_restores_columns_too():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.5, condemn_threshold=0.5)

    reference_col = pool.reference_conductance[:, 3].clone()
    matrix = handle.accessor.read()
    matrix[:, 3] = 999.0
    handle.accessor.write(matrix)

    critical = [(r, 3) for r in range(10)]
    remapped = apply_remap(handle, pool, critical)

    assert remapped == 1
    assert torch.equal(handle.accessor.read()[:, 3], reference_col)

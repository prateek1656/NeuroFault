from neurofault.crossbar.self_healing import RowColumnRedundancyPool
from tests.unit.fakes import make_handle


def test_row_meeting_threshold_is_condemned():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.5, condemn_threshold=0.5)

    # Row 0: 5/10 cells critical -> meets the 0.5 threshold exactly.
    critical = [(0, c) for c in range(5)]
    new_rows, new_cols = pool.condemn_rows_and_cols(critical)

    assert new_rows == [0]
    assert new_cols == []
    assert 0 in pool.condemned_rows


def test_row_below_threshold_is_not_condemned():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.5, condemn_threshold=0.5)

    # Row 0: 4/10 cells critical -> below the 0.5 threshold.
    critical = [(0, c) for c in range(4)]
    new_rows, _new_cols = pool.condemn_rows_and_cols(critical)

    assert new_rows == []
    assert 0 not in pool.condemned_rows


def test_rows_and_columns_have_independent_capacity():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.1, condemn_threshold=0.5)
    assert pool.row_capacity == 1
    assert pool.col_capacity == 1

    # Row 0 fully critical (condemns the row) AND column 5 fully critical
    # (condemns the column) - independent pools, both should succeed.
    critical = [(0, c) for c in range(10)] + [(r, 5) for r in range(10)]
    new_rows, new_cols = pool.condemn_rows_and_cols(critical)

    assert new_rows == [0]
    assert new_cols == [5]
    assert pool.available_row_capacity == 0
    assert pool.available_col_capacity == 0


def test_capacity_exhaustion_prioritizes_most_severe_rows_first():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.1, condemn_threshold=0.5)
    assert pool.row_capacity == 1

    # Row 0: fully critical (10/10). Row 1: just meets threshold (5/10).
    # Only one row-capacity slot - the more severe row must win.
    critical = [(0, c) for c in range(10)] + [(1, c) for c in range(5)]
    new_rows, _new_cols = pool.condemn_rows_and_cols(critical)

    assert new_rows == [0]
    assert 1 not in pool.condemned_rows


def test_already_condemned_row_is_never_reconsidered():
    handle = make_handle(shape=(10, 10))
    pool = RowColumnRedundancyPool(handle, redundancy_factor=0.5, condemn_threshold=0.5)

    critical = [(0, c) for c in range(10)]
    first_rows, _ = pool.condemn_rows_and_cols(critical)
    assert first_rows == [0]

    # Same critical devices again - row 0 must not be returned a second time.
    second_rows, _ = pool.condemn_rows_and_cols(critical)
    assert second_rows == []
    assert len(pool.condemned_rows) == 1

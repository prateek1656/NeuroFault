"""spectral.faults: device-level injection with measured weight-level truth."""

from __future__ import annotations

import pytest
import torch

from neurofault.spectral.faults import inject
from neurofault.spectral.types import (
    ConductanceState,
    EncodingSpec,
    FaultSpec,
    make_generator,
)


def make_state(kind="diff_pair", m=12, n=10, g_min_frac=0.01, seed=0) -> ConductanceState:
    """Hand-built state (no dependency on spectral.encoding)."""
    enc = EncodingSpec(kind=kind, g_min_frac=g_min_frac)
    w = torch.randn(m, n, dtype=torch.float64, generator=make_generator(seed, "w"))
    absmax = float(w.abs().max())
    if kind == "diff_pair":
        s = (enc.g_max_uS - enc.g_min_uS) / absmax
        return ConductanceState(
            enc.g_min_uS + w.clamp(min=0) * s, enc.g_min_uS + (-w).clamp(min=0) * s, s, enc
        )
    if kind == "offset_refcol":
        s = 0.5 * (enc.g_max_uS - enc.g_min_uS) / absmax
        return ConductanceState(0.5 * (enc.g_min_uS + enc.g_max_uS) + w * s, None, s, enc)
    s = enc.g_max_uS / absmax
    return ConductanceState(w * s, None, s, enc)


KINDS = ["stuck_random", "stuck_clustered", "stuck_gradient"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [1, 5, 10, 37])
def test_k_exact_and_cells_unique(kind, k):
    st = make_state()
    new, truth = inject(st, FaultSpec(kind, k=k), make_generator(1, "faults"))
    assert truth.cells.shape == (k, 2)
    assert len({tuple(c) for c in truth.cells.tolist()}) == k
    assert int(new.stuck.sum()) == k  # one device per faulty cell
    assert truth.delta_w.shape == (k,)


def test_density_equals_rounded_k():
    st = make_state()
    _, truth = inject(st, FaultSpec("stuck_random", density=0.1), make_generator(1, "f"))
    assert len(truth.cells) == round(0.1 * 120)


def test_requires_exactly_one_of_k_density():
    with pytest.raises(ValueError):
        inject(make_state(), FaultSpec("stuck_random"), make_generator(1, "f"))
    with pytest.raises(ValueError):
        inject(make_state(), FaultSpec("stuck_random", k=2, density=0.1), make_generator(1, "f"))


def test_input_not_mutated_and_deterministic():
    st = make_state()
    g0 = st.g_pos.clone()
    spec = FaultSpec("stuck_random", k=8)
    a, ta = inject(st, spec, make_generator(7, "faults"))
    b, tb = inject(st, spec, make_generator(7, "faults"))
    _, tc = inject(st, spec, make_generator(8, "faults"))
    assert torch.equal(st.g_pos, g0) and not st.stuck.any()
    assert torch.equal(ta.cells, tb.cells) and torch.equal(a.g_pos, b.g_pos)
    assert not torch.equal(ta.cells, tc.cells)


@pytest.mark.parametrize("rail,expect", [("hrs", "g_min_uS"), ("lrs", "g_max_uS")])
def test_stuck_devices_pinned_to_rail(rail, expect):
    st = make_state()
    new, truth = inject(st, FaultSpec("stuck_random", k=20, rail=rail), make_generator(2, "f"))
    target = getattr(st.enc, expect)
    r, c = truth.cells[:, 0], truth.cells[:, 1]
    vals = torch.where(truth.devices == 0, new.g_pos[r, c], new.g_neg[r, c])
    assert torch.all(vals == target)
    # stuck mask marks exactly the reported devices
    mask = torch.zeros_like(new.stuck)
    mask[r, c, truth.devices] = True
    assert torch.equal(mask, new.stuck)


def test_mixed_rail_uses_both_rails():
    st = make_state(m=20, n=20)
    new, truth = inject(st, FaultSpec("stuck_random", k=100), make_generator(3, "f"))
    r, c = truth.cells[:, 0], truth.cells[:, 1]
    vals = torch.where(truth.devices == 0, new.g_pos[r, c], new.g_neg[r, c])
    assert (vals == st.enc.g_min_uS).any() and (vals == st.enc.g_max_uS).any()


def test_device_selection_pos_neg_either():
    st = make_state()
    _, tp = inject(st, FaultSpec("stuck_random", k=30, device="pos"), make_generator(1, "f"))
    _, tn = inject(st, FaultSpec("stuck_random", k=30, device="neg"), make_generator(1, "f"))
    _, te = inject(st, FaultSpec("stuck_random", k=60, device="either"), make_generator(1, "f"))
    assert (tp.devices == 0).all() and (tn.devices == 1).all()
    assert set(te.devices.tolist()) == {0, 1}
    assert torch.equal(tp.cells, tn.cells)  # same seed -> same fault map irrespective of device


def test_delta_w_is_measured_decoded_change_and_state_dependent():
    st = make_state(g_min_frac=0.0)
    w0 = st.weights()
    for dev in ("pos", "neg"):
        new, truth = inject(
            st, FaultSpec("stuck_random", k=40, rail="hrs", device=dev), make_generator(5, "f")
        )
        r, c = truth.cells[:, 0], truth.cells[:, 1]
        assert torch.allclose(truth.delta_w, (new.weights() - w0)[r, c], atol=1e-14)
        # untouched cells unchanged
        keep = torch.ones_like(w0, dtype=torch.bool)
        keep[r, c] = False
        assert torch.equal(new.weights()[keep], w0[keep])
        # state dependence: hrs on the device carrying the weight removes it, else nothing
        w = w0[r, c]
        carries = (w > 0) if dev == "pos" else (w < 0)
        assert torch.allclose(truth.delta_w[carries], -w[carries], atol=1e-12)
        assert torch.all(truth.delta_w[~carries].abs() < 1e-12)


def test_lrs_delta_w_value():
    st = make_state(g_min_frac=0.0)
    _, truth = inject(
        st, FaultSpec("stuck_random", k=10, rail="lrs", device="pos"), make_generator(5, "f")
    )
    r, c = truth.cells[:, 0], truth.cells[:, 1]
    expected = (st.enc.g_max_uS - st.g_pos[r, c]) / st.scale
    assert torch.allclose(truth.delta_w, expected)


@pytest.mark.parametrize("kind", ["line_row", "line_col"])
def test_line_fault_hits_whole_line(kind):
    st = make_state(m=12, n=10)
    new, truth = inject(st, FaultSpec(kind, k=2, device="pos"), make_generator(4, "f"))
    m, n = st.shape
    if kind == "line_row":
        assert len(truth.rows) == 2 and len(truth.cols) == n
        assert len(truth.cells) == 2 * n
        assert new.stuck[truth.rows][:, :, 0].all()
    else:
        assert len(truth.cols) == 2 and len(truth.rows) == m
        assert len(truth.cells) == 2 * m
        assert new.stuck[:, truth.cols][:, :, 0].all()
    assert int(new.stuck.sum()) == len(truth.cells)


def test_line_open_zeroes_weight_pairs():
    st = make_state()
    new, truth = inject(st, FaultSpec("line_row", k=1, line_mode="open"), make_generator(4, "f"))
    row = int(truth.rows[0])
    assert torch.all(new.weights()[row] == 0)
    assert new.stuck[row].all() and truth.devices is None
    assert torch.allclose(truth.delta_w, -st.weights()[row][truth.cells[:, 1]])


def test_line_density_uses_line_count():
    st = make_state(m=12, n=10)
    _, truth = inject(st, FaultSpec("line_col", density=0.3), make_generator(4, "f"))
    assert len(truth.cols) == 3


def test_clustered_has_spatial_locality():
    st = make_state(m=40, n=40)
    _, tr = inject(st, FaultSpec("stuck_clustered", k=18), make_generator(9, "f"))
    _, tu = inject(st, FaultSpec("stuck_random", k=18), make_generator(9, "f"))

    def spread(t):
        c = t.cells.double()
        return float(c.std(dim=0).sum())

    assert spread(tr) < spread(tu)


def test_offset_refcol_and_signed_ideal():
    for kind in ("offset_refcol", "signed_ideal"):
        st = make_state(kind)
        w0 = st.weights()
        new, truth = inject(st, FaultSpec("stuck_random", k=15, rail="hrs"), make_generator(1, "f"))
        r, c = truth.cells[:, 0], truth.cells[:, 1]
        assert new.g_neg is None and (truth.devices == 0).all()
        assert torch.allclose(truth.delta_w, (new.weights() - w0)[r, c])
        assert int(new.stuck.sum()) == 15
    # offset: hrs -> g_min (weight at the negative extreme); signed: hrs -> 0
    st = make_state("signed_ideal")
    new, truth = inject(st, FaultSpec("stuck_random", k=5, rail="hrs"), make_generator(1, "f"))
    assert torch.all(new.weights()[truth.cells[:, 0], truth.cells[:, 1]] == 0)
    st = make_state("offset_refcol")
    new, truth = inject(st, FaultSpec("stuck_random", k=5, rail="hrs"), make_generator(1, "f"))
    assert torch.all(new.g_pos[truth.cells[:, 0], truth.cells[:, 1]] == st.enc.g_min_uS)


def test_neg_device_rejected_without_pair():
    with pytest.raises(ValueError):
        inject(
            make_state("offset_refcol"),
            FaultSpec("stuck_random", k=2, device="neg"),
            make_generator(1, "f"),
        )


def test_same_fault_map_across_encodings():
    spec = FaultSpec("stuck_random", k=12)
    cells = [
        inject(make_state(kind), spec, make_generator(3, "faults"))[1].cells
        for kind in ("diff_pair", "offset_refcol", "signed_ideal")
    ]
    assert torch.equal(cells[0], cells[1]) and torch.equal(cells[0], cells[2])

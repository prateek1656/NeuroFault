import torch

from neurofault.mitigation.reset import apply_layer_reset
from tests.unit.fakes import FakeCrossbar


def _fake_double_column_map(weight, r_on, r_off, scheme):
    # Mimic naive_map: DoubleColumn -> (positive, negative) tensors; SingleColumn -> one tensor.
    if scheme == "DoubleColumn":
        return (torch.ones_like(weight) / r_on, torch.ones_like(weight) / r_off)
    return torch.ones_like(weight) / r_on


def test_layer_reset_restores_weights_and_rewrites_every_handle():
    layer = torch.nn.Linear(4, 4)
    original_weight = layer.weight.data.clone()
    layer.weight.data.fill_(999.0)  # simulate drifted/corrupted weights

    handles = [
        type("H", (), {"accessor": FakeCrossbar((4, 4))})(),
        type("H", (), {"accessor": FakeCrossbar((4, 4))})(),
    ]

    ok = apply_layer_reset(
        layer,
        handles,
        original_weight,
        _fake_double_column_map,
        r_on=100.0,
        r_off=10000.0,
        scheme="DoubleColumn",
    )

    assert ok is True
    assert torch.equal(layer.weight.data, original_weight)
    assert torch.allclose(handles[0].accessor.read(), torch.ones(4, 4) / 100.0)
    assert torch.allclose(handles[1].accessor.read(), torch.ones(4, 4) / 10000.0)


def test_layer_reset_returns_false_on_failure_never_silently_succeeds():
    """Direct regression test for bug #5: the original swallowed the
    exception in a bare `except Exception: print(...)` and the caller had
    no way to tell reset had failed."""
    layer = torch.nn.Linear(4, 4)
    original_weight = layer.weight.data.clone()

    # Mismatched handle count vs. what the mapping routine produces -> must fail loudly.
    handles = [type("H", (), {"accessor": FakeCrossbar((4, 4))})()]

    ok = apply_layer_reset(
        layer,
        handles,
        original_weight,
        _fake_double_column_map,
        r_on=100.0,
        r_off=10000.0,
        scheme="DoubleColumn",
    )

    assert ok is False

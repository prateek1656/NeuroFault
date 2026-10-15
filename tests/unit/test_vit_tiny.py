import pytest
import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import ViTTiny, build_model
from neurofault.retraining.bn_recal import recalibrate_bn


def _model(**kw):
    return build_model(ExperimentConfig(name="v", model_arch="vit_tiny", **kw))


def test_vit_tiny_builds_and_has_expected_shape():
    m = _model().eval()
    assert type(m) is ViTTiny
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 <= n <= 1_400_000
    names = [k for k, x in m.named_modules() if isinstance(x, torch.nn.Linear)]
    assert len(names) == 1 + 4 * 4 + 1 and names[-1] == "fc"  # embed + 4x(qkv,proj,fc1,fc2) + head
    assert not any(isinstance(x, torch.nn.Conv2d) for x in m.modules())
    assert not any(isinstance(x, torch.nn.MultiheadAttention) for x in m.modules())


def test_vit_tiny_c100_head_and_state_dict_roundtrip():
    m = _model(dataset="cifar100", num_classes=100)
    assert m.fc.out_features == 100
    ViTTiny().load_state_dict(ViTTiny().state_dict(), strict=True)


def test_every_linear_sees_2d_input():
    # CrossSim AnalogLinear backward mis-shapes the bias gradient for 3-D input.
    m = _model().eval()
    seen = []
    for x in m.modules():
        if isinstance(x, torch.nn.Linear):
            x.register_forward_hook(lambda _m, inp, _o: seen.append(inp[0].ndim))
    m(torch.randn(2, 3, 32, 32))
    assert seen and set(seen) == {2}


def test_bn_recal_is_safe_noop_without_batchnorm():
    m = _model()
    assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
    m.train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    loader = [(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))] * 3
    assert recalibrate_bn(m, loader, 20, "cpu") == 0
    assert m.training  # train/eval flags untouched
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_gradients_reach_every_parameter():
    m = _model().train()
    m(torch.randn(4, 3, 32, 32)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_adamw_option_accepted_and_bad_name_rejected():
    ExperimentConfig(name="x", model_arch="vit_tiny", optimizer="adamw")
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", optimizer="lion")


def test_patching_with_crosssim_if_available():
    pytest.importorskip("simulator")
    from simulator.algorithms.dnn.torch.linear import AnalogLinear

    from neurofault.crossbar.backends import crosssim_backend as cb

    m = _model().eval()
    ref = m(torch.randn(2, 3, 32, 32)).shape
    seen = [
        t[0]
        for t in cb._walk_patch_targets(
            m, (torch.nn.Linear, torch.nn.Conv2d), {torch.nn.Linear: AnalogLinear}
        )
    ]
    assert len(seen) == 18 and ref == (2, 10)

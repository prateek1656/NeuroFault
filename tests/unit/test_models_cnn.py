import torch

from neurofault.models.cnn import SimpleCNN


def test_forward_pass_shape():
    model = SimpleCNN()
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    out = model(x)

    assert out.shape == (2, 10)


def test_build_model_dispatches_on_model_arch_and_rejects_unknown():
    import pytest

    from neurofault.config import ExperimentConfig
    from neurofault.models.cnn import TinyCNN, build_model

    assert type(build_model(ExperimentConfig(name="x"))) is SimpleCNN  # default stays resnet9
    assert type(build_model(ExperimentConfig(name="x", model_arch="resnet9"))) is SimpleCNN
    assert type(build_model(ExperimentConfig(name="x", model_arch="tiny_cnn"))) is TinyCNN
    with pytest.raises(ValueError, match="model_arch"):
        ExperimentConfig(name="x", model_arch="vgg")


def test_tiny_cnn_is_small_cifar_shaped_and_shares_simplecnn_conventions():
    from neurofault.models.cnn import TinyCNN

    model = TinyCNN().eval()
    n_weights = sum(p.numel() for p in model.parameters())
    assert 10_000 <= n_weights <= 50_000
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    # same module kinds as SimpleCNN: bias-free convs each followed by BN, one Linear
    assert all(m.bias is None for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(convs) == len(bns) >= 3
    assert [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)] == ["fc"]
    # state dicts round-trip through a fresh instance of the same arch (checkpoint load path)
    TinyCNN().load_state_dict(model.state_dict(), strict=True)

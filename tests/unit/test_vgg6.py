import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import VGG6, build_model


def test_vgg6_builds_and_has_expected_shape():
    m = build_model(ExperimentConfig(name="v", model_arch="vgg6")).eval()
    assert type(m) is VGG6
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert 1_000_000 <= sum(p.numel() for p in m.parameters()) <= 1_300_000
    convs = [x for x in m.modules() if isinstance(x, torch.nn.Conv2d)]
    assert len(convs) == 6 and all(c.bias is None for c in convs)
    assert [n for n, x in m.named_modules() if isinstance(x, torch.nn.Linear)] == ["fc"]
    assert not any(hasattr(m, a) for a in ("res1", "res2"))  # no residual blocks


def test_vgg6_c100_head_and_state_dict_roundtrip():
    m = build_model(
        ExperimentConfig(name="v", model_arch="vgg6", dataset="cifar100", num_classes=100)
    )
    assert m.fc.out_features == 100
    VGG6().load_state_dict(VGG6().state_dict(), strict=True)

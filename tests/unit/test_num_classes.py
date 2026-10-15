import pytest
import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import build_model


@pytest.mark.parametrize("arch", ["tiny_cnn", "resnet9"])
def test_default_head_is_10_and_c100_head_is_100(arch):
    m10 = build_model(ExperimentConfig(name="a", model_arch=arch))
    m100 = build_model(
        ExperimentConfig(name="b", model_arch=arch, dataset="cifar100", num_classes=100)
    )
    assert m10.fc.out_features == 10 and m100.fc.out_features == 100
    # same state_dict keys: CIFAR-10 checkpoints load unchanged
    assert m10.state_dict().keys() == m100.state_dict().keys()


def test_tiny_cnn_c100_forward_shape():
    m = build_model(
        ExperimentConfig(name="b", model_arch="tiny_cnn", dataset="cifar100", num_classes=100)
    )
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 100)


def test_num_classes_must_match_dataset():
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", dataset="cifar100")
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", dataset="cifar10", num_classes=100)

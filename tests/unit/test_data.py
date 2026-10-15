import pytest

from neurofault.config import ExperimentConfig
from neurofault.data import load_datasets


@pytest.mark.parametrize("dataset", ["cifar10", "mnist"])
def test_load_datasets_yields_correctly_shaped_batches(dataset):
    config = ExperimentConfig(
        name=f"data_check_{dataset}",
        dataset=dataset,
        train_samples=16,
        test_samples=8,
        batch_size=4,
    )

    trainloader, _testloader = load_datasets(config)
    x, y = next(iter(trainloader))

    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)


def test_load_datasets_rejects_unknown_dataset():
    """ExperimentConfig.__post_init__ now fails this fast, at construction -
    load_datasets()'s own `else: raise ValueError` is unreachable via a
    validated config, but stays as defense-in-depth (see config.py)."""
    with pytest.raises(ValueError, match="Unknown ExperimentConfig.dataset"):
        ExperimentConfig(name="bad_dataset", dataset="not_a_real_dataset")

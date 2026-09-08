import pytest

from retention_risk.data import build_dataset


@pytest.fixture(scope="session")
def dataset():
    """The default modelling dataset, built once per test session."""
    return build_dataset()

from __future__ import annotations

from pathlib import Path

import pytest

from t4_devkit import T4Devkit


@pytest.fixture(scope="session")
def sample_dataset_path() -> Path:
    return Path(__file__).parents[1] / "sample/t4dataset"


@pytest.fixture(scope="session")
def t4(sample_dataset_path: Path) -> T4Devkit:
    """Return a T4Devkit instance loaded from the sample dataset."""
    return T4Devkit(sample_dataset_path, verbose=False)

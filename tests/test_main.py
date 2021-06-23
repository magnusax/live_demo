# fmt: off
import sys
import pytest
from pathlib import Path

root_folder = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_folder)

from main import get_data  # noqa: E402

@pytest.fixture(scope="module")
def dataset():
    X, labels = get_data()
    return X

def test_dataset_shapes(dataset):
    assert len(dataset.shape) == 2
    assert dataset.shape[0] == 750

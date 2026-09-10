import random

import numpy as np

from aroma.utils import set_seed


def test_seed_reproducibility():
    set_seed(42)

    a = random.random()
    b = np.random.rand()

    set_seed(42)

    assert random.random() == a
    assert np.random.rand() == b

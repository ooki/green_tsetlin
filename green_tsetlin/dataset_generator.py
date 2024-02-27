
from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState



def xor_dataset(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_literals: int = 4,
                seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate a dataset where y = x[0] XOR x[1], with the rest of the features being dummy variables.
    :param noise:   the probability for a noisy training example
    :param n_train: number of training examples
    :param n_test: number of test examples
    :param n_literals: the number of literals, must be at least 2.
    :param seed: random seed. default: None
    :return: train_x (uint8), train_y (uint32), test_x (uint8), test_y (uint32)
    """
    prng = RandomState(seed)

    if n_literals < 2:
        raise ValueError(F"Cannot create xor dataset with less than 2 literals ({n_literals} specified)")

    train_x = prng.randint(low=0, high=2, size=(n_train, n_literals))
    train_y = np.logical_xor(train_x[:, 0], train_x[:, 1])

    if noise is not None and noise > 0.0:
        flips = prng.random(n_train) < noise
        train_y[flips] = np.logical_not(train_y[flips])

    test_x = prng.randint(low=0, high=2, size=(n_test, n_literals))
    test_y = np.logical_xor(test_x[:, 0], test_x[:, 1])

    return train_x.astype(np.uint8), train_y.astype(np.uint32), test_x.astype(np.uint8), test_y.astype(np.uint32)

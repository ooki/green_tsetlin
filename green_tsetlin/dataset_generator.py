


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

    oy = np.copy(train_y)

    if noise is not None and noise > 0.0:
        flips = prng.random(n_train) < noise
        train_y[flips] = np.logical_not(train_y[flips])

    test_x = prng.randint(low=0, high=2, size=(n_test, n_literals))
    test_y = np.logical_xor(test_x[:, 0], test_x[:, 1])

    return train_x.astype(np.uint8), train_y.astype(np.uint32), test_x.astype(np.uint8), test_y.astype(np.uint32)



def multi_label_xor(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_literals: int = 4, n_classes:int = 3,
                        seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate a dataset where:
        y0 = x[0] XOR x[1]
        y1 = x[0] XOR x[2]
        y2 = x[0] XOR x[3]
        ..        
        with the rest of the features being dummy variables.

    :param noise:   the probability for a noisy training example
    :param n_train: number of training examples
    :param n_test: number of test examples
    :param n_literals: the number of literals, must be at least 2.
    :param n_classes: number of classes to generate
    :param seed: random seed. default: None
    :return: train_x (uint8), train_y (uint32), test_x (uint8), test_y (uint32)
    """
    prng = RandomState(seed)

    if n_literals < (1 + n_classes):
        raise ValueError(F"Cannot create multilabel xor dataset with less than {(1 + n_classes)} literals ({n_literals} specified)")

    train_x = prng.randint(low=0, high=2, size=(n_train, n_literals))
    train_ys = []
    for k in range(0, n_classes):
        class_y = np.logical_xor(train_x[:, 0], train_x[:, k+1])
        train_ys.append(class_y[:, np.newaxis])
    train_y = np.concatenate(train_ys, axis=1)
        


    if noise is not None and noise > 0.0:
        flips = prng.random(size=(train_y.shape)) < noise        
        train_y[flips] = np.logical_not(train_y[flips])

    test_x = prng.randint(low=0, high=2, size=(n_test, n_literals))
    test_ys = []
    for k in range(0, n_classes):
        class_y = np.logical_xor(test_x[:, 0], test_x[:, k+1])
        test_ys.append(class_y[:, np.newaxis])
    test_y = np.concatenate(test_ys, axis=1)
    
    return train_x.astype(np.uint8), train_y.astype(np.uint32), test_x.astype(np.uint8), test_y.astype(np.uint32)



def xor_dataset_with_negated(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_literals: int = 4,
                seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate a dataset where y = x[0] XOR x[1], with the rest of the features being dummy variables.
        Will concat an negated version at the end so that it is posible to learn without negated TA's.s
    :param noise:   the probability for a noisy training example
    :param n_train: number of training examples
    :param n_test: number of test examples
    :param n_literals: the number of literals, must be at least 4 (to fit: 2 positive, 2 negated), and even.
    :param seed: random seed. default: None
    :return: train_x (uint8), train_y (uint32), test_x (uint8), test_y (uint32)
    """


    if n_literals < 4:
        raise ValueError(F"Cannot create negated xor dataset with less than 4 literals ({n_literals} specified)")

    if n_literals % 2 != 0:
        raise ValueError(F"Cannot create negated xor dataset with odd numbered literals ({n_literals} specified)")

    train_x, train_y, test_x, test_y = xor_dataset(noise, n_train, n_test, n_literals // 2, seed)

    return np.hstack([train_x, np.logical_not(train_x)]), train_y, np.hstack([test_x, np.logical_not(test_x)]), test_y


def xor_dataset_with_negated_sparse(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_literals: int = 4, active_literals: int = 4,
                seed: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate a dataset where y = x[0] XOR x[1], with the rest of the features being dummy variables.
        Will concat an negated version at the end so that it is posible to learn without negated TA's.s
    :param noise:   the probability for a noisy training example
    :param n_train: number of training examples
    :param n_test: number of test examples
    :param n_literals: the number of literals, must be at least 4 (to fit: 2 positive, 2 negated), and even.
    :param seed: random seed. default: None
    :return: train_x (uint8), train_y (uint32), test_x (uint8), test_y (uint32)
    """

    if active_literals < 4:
        raise ValueError(F"Cannot create negated xor dataset with less than 4 literals active ({active_literals} specified)")
    
    if n_literals < active_literals:
        raise ValueError(F"Cannot create negated xor dataset with less literals than active literals ({n_literals} specified)")

    if active_literals % 2 != 0:
        raise ValueError(F"Cannot create negated xor dataset with odd numbered active_literals ({active_literals} specified)")

    train_x, train_y, test_x, test_y = xor_dataset(noise, n_train, n_test, active_literals // 2, seed)
    
    sparse_x_train = np.zeros(shape=(n_train, n_literals - active_literals), dtype=np.uint8)
    sparse_x_test = np.zeros(shape=(n_test, n_literals - active_literals), dtype=np.uint8)

    return np.hstack([train_x, np.logical_not(train_x), sparse_x_train]), train_y, np.hstack([test_x, np.logical_not(test_x), sparse_x_test]), test_y






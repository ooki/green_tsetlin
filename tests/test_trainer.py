from collections import namedtuple

import pytest

import numpy as np

import green_tsetlin as gt


def test_trainer_throws_if_data_is_wrong_dtype():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23
    threshold = 42
    x = np.ones([2, n_literals], dtype=np.uint8)
    y = np.ones([0, 1], dtype=np.uint32)

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)

    trainer = gt.Trainer(tm)
    trainer.set_data(x, y)
    trainer.set_data(x, y, x, y)

    x_wrong = np.ones([2, n_literals], dtype=np.int8)
    y_wrong = np.ones([2], dtype=np.int32)
    with pytest.raises(ValueError):
        trainer.set_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_data(x, y_wrong)    

    with pytest.raises(ValueError):
        trainer.set_data(x, y, x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_data(x, y, x, y_wrong)
    
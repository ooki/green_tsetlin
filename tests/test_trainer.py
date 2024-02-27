from collections import namedtuple

import pytest

import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc


def test_trainer_throws_if_data_is_wrong_dtype():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23
    threshold = 42
    x = np.ones((2, n_literals), dtype=np.uint8)
    y = np.ones(2, dtype=np.uint32)
    x_wrong = np.ones((2, n_literals), dtype=np.int8)
    y_wrong = np.ones(2, dtype=np.int32)

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)

    trainer = gt.Trainer(tm)
    trainer.set_train_data(x, y)
    trainer.set_test_data(x, y)
    
    with pytest.raises(ValueError):
        trainer.set_train_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_train_data(x, y_wrong)

    with pytest.raises(ValueError):
        trainer.set_test_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_test_data(x, y_wrong)

    

def test_trainer_throws_on_wrong_number_of_examples_between_x_and_y():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23
    threshold = 42
    x = np.ones((2, n_literals), dtype=np.uint8)
    y = np.ones(2, dtype=np.uint32)
    x_wrong = np.ones((1, n_literals), dtype=np.uint8)
    y_wrong = np.ones(1, dtype=np.uint32)

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)

    trainer = gt.Trainer(tm)
    trainer.set_train_data(x, y)
    trainer.set_test_data(x, y)

    with pytest.raises(ValueError):
        trainer.set_train_data(x, y_wrong)

    with pytest.raises(ValueError):
        trainer.set_test_data(x, y_wrong)        
    
    with pytest.raises(ValueError):
        trainer.set_train_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_test_data(x_wrong, y)


def test_train_simple_xor():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    r = trainer.train()    
    print(r)
    


if __name__ == "__main__":
    test_trainer_throws_on_wrong_number_of_examples_between_x_and_y()
    test_train_simple_xor()
    
    print("<done: ", __file__, ">")
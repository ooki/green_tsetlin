from collections import namedtuple

import pytest

import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc
import green_tsetlin.py_gtc as py_gtc


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
    
    
def test_train_simple_xor_backend():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM
    print("BACKEND:", tm._backend_clause_block_cls)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    r = trainer.train()    
    print(r)
    
    
def test_train_set_best_state_afterwards():    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._backend_clause_block_cls = gtc.ClauseBlockTM

    assert tm._state is None
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, load_best_state=True, progress_bar=False, n_epochs=3)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    trainer.train()    

    assert tm._state is not None
    
def test_pygtc_backend_set():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    
    tm._backend_clause_block_cls = py_gtc.ClauseBlock
    trainer._cls_feedback_block = py_gtc.FeedbackBlock
    trainer._cls_dense_ib = py_gtc.DenseInputBlock
    trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor


    print("BACKEND:")
    print(tm._backend_clause_block_cls)
    print(trainer._cls_feedback_block)
    print(trainer._cls_dense_ib)
    print(trainer._cls_exec_singlethread)

    # TO DO: executor train_epoch -> train_slice

if __name__ == "__main__":
    #test_trainer_throws_on_wrong_number_of_examples_between_x_and_y()
    #test_train_simple_xor()
    #test_train_set_best_state_afterwards()
    test_pygtc_backend_set()
    print("<done: ", __file__, ">")
from collections import namedtuple

import pytest

import numpy as np

from scipy.sparse import csr_matrix

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
    trainer.train()    
    
    
    
    
    
    
    
def test_train_simple_xor_gtc_tm_backend():
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=40)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    r = trainer.train()    
    assert r["did_early_exit"]
    
    
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
    
def test_train_simple_xor_py_gtc():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42   
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    tm._backend_clause_block_cls = py_gtc.ClauseBlock

    trainer = gt.Trainer(tm, seed=32, n_jobs=1 )

    trainer._cls_feedback_block = py_gtc.FeedbackBlock
    trainer._cls_dense_ib = py_gtc.DenseInputBlock
    trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor

    
    print("BACKEND:")
    print(tm._backend_clause_block_cls)
    print(tm._backend_clause_block_cls)
    print(trainer._cls_feedback_block)
    print(trainer._cls_dense_ib)
    print(trainer._cls_exec_singlethread)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    r = trainer.train()    
    print(r)

def test_select_backend_ib():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    tm._backend_clause_block_cls = gtc.ClauseBlockTM

    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    sparse_x = csr_matrix(x)
    sparse_ex = csr_matrix(ex)
    
    # test both dense
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    trainer._select_backend_ib()
    assert trainer._cls_input_block == gtc.DenseInputBlock, (trainer._cls_input_block, trainer._cls_dense_ib)

    # test both sparse
    trainer.set_train_data(sparse_x, y)
    trainer.set_test_data(sparse_ex, ey)
    trainer._select_backend_ib()
    assert trainer._cls_input_block == gtc.SparseInputBlock, (trainer._cls_input_block, trainer._cls_sparse_ib)

    # test train dense, test sparse
    trainer.set_train_data(x, y)
    trainer.set_test_data(sparse_ex, ey)
    with pytest.raises(ValueError):
        trainer._select_backend_ib()

    # test train sparse, test dense
    trainer.set_train_data(sparse_x, y)
    trainer.set_test_data(ex, ey)
    with pytest.raises(ValueError):
        trainer._select_backend_ib()
    



def test_train_simple_xor_sparse():
    print("SPARSE\n")
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    tm._backend_clause_block_cls = gtc.ClauseBlockSparse

    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    
    print("BACKEND:")
    print(tm._backend_clause_block_cls)
    print(trainer._cls_feedback_block)
    print(trainer._cls_dense_ib)
    print(trainer._cls_sparse_ib)
    print(trainer._cls_exec_singlethread)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    

    x = csr_matrix(x)
    ex = csr_matrix(ex)

    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)

    r = trainer.train()    
    print(r)


def test_set_backend_py_gtc_sparse(): # Should be one test in the future
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        

    tm._backend_clause_block_cls = py_gtc.ClauseBlockSparse
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)

    trainer._cls_feedback_block = py_gtc.FeedbackBlock
    trainer._cls_sparse_ib = py_gtc.SparseInputBlock
    trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor

    print("BACKEND:")
    print(tm._backend_clause_block_cls)
    print(trainer._cls_feedback_block)
    print(trainer._cls_sparse_ib)
    print(trainer._cls_exec_singlethread)
    
    # x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    

    # x = csr_matrix(x)
    # ex = csr_matrix(ex)

    # trainer.set_train_data(x, y)
    # trainer.set_test_data(ex, ey)

    # r = trainer.train()    
    # print(r)


if __name__ == "__main__":
    #test_trainer_throws_on_wrong_number_of_examples_between_x_and_y()
    #sstest_train_simple_xor()
    #test_train_set_best_state_afterwards()
    # test_train_simple_xor_py_gtc()
    test_train_simple_xor_sparse()
    # test_train_simple_xor_gtc_tm_backend()
    # test_select_backend_ib()
    # test_set_backend_py_gtc_sparse()
    print("<done: ", __file__, ">")
from collections import namedtuple

import pytest

import numpy as np

# import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

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
    trainer.set_eval_data(x, y)
    
    with pytest.raises(ValueError):
        trainer.set_train_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_train_data(x, y_wrong)

    with pytest.raises(ValueError):
        trainer.set_eval_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_eval_data(x, y_wrong)

    
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
    trainer.set_eval_data(x, y)

    with pytest.raises(ValueError):
        trainer.set_train_data(x, y_wrong)

    with pytest.raises(ValueError):
        trainer.set_eval_data(x, y_wrong)        
    
    with pytest.raises(ValueError):
        trainer.set_train_data(x_wrong, y)

    with pytest.raises(ValueError):
        trainer.set_eval_data(x_wrong, y)


def test_train_simple_xor():
    
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4, boost_true_positives=False)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)

    trainer.train()    
    # print("BACKEND:")
    # print(tm._backend_clause_block_cls)


def test_train_simple_xor_consistency():
    
    train_logs = []
    best_accs = []
    seed = 42
    other_seed = 43
    for i in range(4):

        n_literals = 6
        n_clauses = 5
        n_classes = 2
        s = 3.0
        threshold = 42    
        tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
        #tm._backend_clause_block_cls = gtc.ClauseBlockTM

        x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, seed=seed)    
        if i == 3:
            seed = other_seed
        trainer = gt.Trainer(tm, seed=seed, n_jobs=1, progress_bar=False, n_epochs=2)
        trainer.set_train_data(x, y)
        trainer.set_eval_data(ex, ey)
        trainer.train()

        train_logs.append(trainer.results["train_log"])
        best_accs.append(trainer.results["best_eval_score"])


    assert np.array_equal(train_logs[0], train_logs[1])
    assert np.array_equal(train_logs[1], train_logs[2])
    assert np.array_equal(best_accs[0], best_accs[1])
    assert np.array_equal(best_accs[1], best_accs[2])

    assert not np.array_equal(train_logs[0], train_logs[-1])
    assert not np.array_equal(best_accs[0], best_accs[-1])


def test_train_simple_xor_consistency_sparse():
    
    train_logs = []
    best_accs = []
    seed = 42
    other_seed = 44
    for i in range(4):

        n_literals = 6
        n_clauses = 5
        n_classes = 2
        s = 3.0
        threshold = 42    
        tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4, boost_true_positives=True, dynamic_AL=True)        
        # #tm._backend_clause_block_cls = gtc.ClauseBlockTM

        x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, seed=seed)    
        if i == 3:
            seed = other_seed
        trainer = gt.Trainer(tm, seed=seed, n_jobs=1, progress_bar=False, load_best_state=False, n_epochs=2)
        trainer.set_train_data(csr_matrix(x), y)
        trainer.set_eval_data(csr_matrix(ex), ey)
        trainer.train()

        train_logs.append(trainer.results["train_log"])
        best_accs.append(trainer.results["best_eval_score"])


    assert np.array_equal(train_logs[0], train_logs[1])
    assert np.array_equal(train_logs[1], train_logs[2])
    assert np.array_equal(best_accs[0], best_accs[1])
    assert np.array_equal(best_accs[1], best_accs[2])

    assert not np.array_equal(train_logs[0], train_logs[-1]), (train_logs[0], train_logs[-1])
    assert not np.array_equal(best_accs[0], best_accs[-1]), (best_accs[0], best_accs[-1])


def test_train_simple_xor_uniform_feedback():
    
    n_literals = 6
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM

    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, feedback_type="uniform", n_epochs=100)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    trainer.train()    

    assert trainer.results["did_early_exit"]
    

    
def test_train_simple_xor_gtc_tm_backend():
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=40)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()    
    
    assert r["did_early_exit"]
    assert sum(r["train_time_of_epochs"]) > 0.000001        


def test_train_set_best_state_and_results_afterwards():    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    assert tm._state is None
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, load_best_state=True, progress_bar=False, n_epochs=3)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)

    assert trainer.results is None
    trainer.train()    

    assert trainer.results is not None
    assert tm._state is not None

def mock_return_gtc_backend():
    return py_gtc.ClauseBlock

def mock_return_gtc_tm_backend_sparse_input_dense_output():
    return py_gtc.SparseInpuDenseOutputBlock

def test_train_simple_xor_py_gtc():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42   
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._get_backend = mock_return_gtc_backend
    

    trainer = gt.Trainer(tm, seed=32, n_jobs=1 )

    trainer._cls_feedback_block = py_gtc.FeedbackBlock
    trainer._cls_dense_ib = py_gtc.DenseInputBlock
    trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor

    

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()    
    print(r)

def test_train_simple_xor_sparse_input_dense_backend_pygtc():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42   
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._get_backend = mock_return_gtc_backend
    

    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=100)

    trainer._cls_feedback_block = py_gtc.FeedbackBlock    
    trainer._cls_sparse_input_dense_output_ib = py_gtc.SparseInpuDenseOutputBlock
    trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor

    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, noise=0.05)    
    sparse_x = csr_matrix(x)
    sparse_ex = csr_matrix(ex)

    trainer.set_train_data(sparse_x, y)
    trainer.set_eval_data(sparse_ex, ey)
    r = trainer.train()    
    assert r["best_eval_score"] > 0.99

def test_train_simple_xor_sparse_input_dense_backend_gtc():
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42   
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=100, progress_bar=False)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, noise=0.05)    
    sparse_x = csr_matrix(x)
    sparse_ex = csr_matrix(ex)

    trainer.set_train_data(sparse_x, y)
    trainer.set_eval_data(sparse_ex, ey)
    r = trainer.train()    
    assert r["best_eval_score"] > 0.99




def test_select_backend_ib_trainer_dense():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    sparse_x = csr_matrix(x)
    sparse_ex = csr_matrix(ex)
    
    # test both dense
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    trainer._select_backend_ib()
    assert trainer._cls_input_block == gtc.DenseInputBlock, (trainer._cls_input_block, trainer._cls_dense_ib)


    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(sparse_x, y)
    trainer.set_eval_data(sparse_ex, ey)
    trainer._select_backend_ib()
    assert trainer._cls_input_block == gtc.SparseInputDenseOutputBlock, (trainer._cls_input_block, trainer._cls_dense_ib)


    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(sparse_x, y)

    # test train spase, test dense
    with pytest.raises(ValueError):
        trainer.set_eval_data(x, ey)

    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)

    # test train dense, test sparse
    with pytest.raises(ValueError):
        trainer.set_eval_data(sparse_x, y)
    

def test_select_backend_ib_trainer_sparse():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    

    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    sparse_x = csr_matrix(x)
    sparse_ex = csr_matrix(ex)
    
    # test both sparse
    trainer.set_train_data(sparse_x, y)
    trainer.set_eval_data(sparse_ex, ey)
    trainer._select_backend_ib()
    assert trainer._cls_input_block == gtc.SparseInputBlock, (trainer._cls_input_block, trainer._cls_sparse_ib)

    # test train dense, test sparse
    with pytest.raises(ValueError):
        trainer.set_eval_data(ex, ey)

    # test train sparse, test dense
    with pytest.raises(ValueError):
        trainer.set_train_data(x, y)


def test_train_simple_xor_sparse():    
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4, boost_true_positives=False, dynamic_AL=False)        
    
    # tm._backend_clause_block_cls = gtc.ClauseBlockSparse_Lt_Dt_Bt
    tm.active_literals_size = n_literals
    tm.clause_size = n_literals
    tm.lower_ta_threshold = -40
    # tm.set_dynamic_AL(True)

    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=40, load_best_state=False)
    

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    # seed=6

    x = csr_matrix(x)
    ex = csr_matrix(ex)

    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)

    r = trainer.train()    
    # print("BACKEND:")
    # print(tm._backend_clause_block_cls)
    # print(trainer._cls_feedback_block)
    # print(trainer._cls_dense_ib)
    # print(trainer._cls_sparse_ib)
    # print(trainer._cls_exec_singlethread)
    # print(r)


# def test_set_backend_py_gtc_sparse(): # Should be one test in the future
#     n_literals = 7
#     n_clauses = 5
#     n_classes = 2
#     s = 3.0
#     threshold = 42    
#     tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
#     tm._get_backend = mock_return_gtc_backend
#     tm._backend_clause_block_cls = py_gtc.ClauseBlockSparse
#     trainer = gt.Trainer(tm, seed=32, n_jobs=1)

#     trainer._cls_feedback_block = py_gtc.FeedbackBlock
#     trainer._cls_sparse_ib = py_gtc.SparseInputBlock
#     trainer._cls_exec_singlethread = py_gtc.SingleThreadExecutor

#     print("BACKEND:")
#     print(tm._backend_clause_block_cls)
#     print(trainer._cls_feedback_block)
#     print(trainer._cls_sparse_ib)
#     print(trainer._cls_exec_singlethread)
    
#     x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    

#     x = csr_matrix(x)
#     ex = csr_matrix(ex)

#     trainer.set_train_data(x, y)
#     trainer.set_eval_data(ex, ey)

#     r = trainer.train()    
#     print(r)


def test_trainer_with_kfold():

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    #tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=False, k_folds=20, kfold_progress_bar=True)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()

    assert r["best_eval_score"] == 1.0


def test_wrong_data_formats():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=False, k_folds=20, kfold_progress_bar=True)

    # with pytest.raises(ValueError):
    #     trainer.set_train_data(csr_matrix(x), y)

    # with pytest.raises(ValueError):
    #     trainer.set_eval_data(csr_matrix(ex), y)

    # with pytest.raises(ValueError):
    #     trainer.set_validation_data(csr_matrix(ex), y)


    tm_s = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)
    trainer_s = gt.Trainer(tm_s, seed=32, n_jobs=1, progress_bar=False, k_folds=20, kfold_progress_bar=True)

    with pytest.raises(ValueError):
        trainer_s.set_train_data(x, y)

    with pytest.raises(ValueError):
        trainer_s.set_eval_data(ex, y)



def test_trainer_save_last_state_if_save_best_is_false():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=True, load_best_state=False)

    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()

    # print(r)

    assert r["best_eval_score"] == 1.0
    assert tm._state is not None
    assert trainer._best_tm_state is None


def test_trainer_save_best_state_if_save_best_is_true():
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=True, load_best_state=True)

    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()

    # print(r)

    assert r["best_eval_score"] == 1.0
    assert tm._state is not None
    assert trainer._best_tm_state is not None
    assert tm._state == trainer._best_tm_state


def test_set_test_train():

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    train_x, train_y, test_x, test_y = gt.dataset_generator.xor_dataset(n_literals=n_literals)    

    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=False)

    trainer.set_train_data(train_x, train_y)
    trainer.set_eval_data(test_x, test_y)
    
    assert trainer.x_train.shape == train_x.shape
    assert trainer.x_eval.shape == test_x.shape
    


if __name__ == "__main__":
    # test_trainer_throws_on_wrong_number_of_examples_between_x_and_y()
    # test_train_set_best_state_and_results_afterwards()
    # test_train_simple_xor_py_gtc()
    # test_train_simple_xor_sparse()
    # test_train_simple_xor()
    # test_train_simple_xor_gtc_tm_backend()
    # test_select_backend_ib()
    # test_set_backend_py_gtc_sparse()
    # test_wrong_data_formats()
    # test_train_simple_xor_consistency()
    # test_train_simple_xor_consistency_sparse()

    # test_train_simple_xor_uniform_feedback()

    # test_trainer_with_kfold()

    # test_trainer_save_last_state_if_save_best_is_false()
    # test_trainer_save_best_state_if_save_best_is_true()
    # test_train_simple_xor_sparse_input_dense_backend_gtc()

    test_set_test_train()
    test_train_simple_xor_sparse_input_dense_backend_gtc()
    test_set_test_val_train()

    print("<done: ", __file__, ">")
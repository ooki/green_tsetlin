from collections import namedtuple

import pytest

import numpy as np

import datasets
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
    
    n_literals = 15
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
    assert sum(r["train_time_of_epochs"]) > 0.000001        


def test_train_set_best_state_and_results_afterwards():    
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

    assert trainer.results is None
    trainer.train()    

    assert trainer.results is not None
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

    
    # print("BACKEND:")
    # print(tm._backend_clause_block_cls)
    # print(tm._backend_clause_block_cls)
    # print(trainer._cls_feedback_block)
    # print(trainer._cls_dense_ib)
    # print(trainer._cls_exec_singlethread)

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
    n_literals = 15
    n_clauses = 50
    n_classes = 2
    s = 3.0
    threshold = 42
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    
    tm._backend_clause_block_cls = gtc.ClauseBlockSparse
    tm.set_active_literals_size(8)
    tm.set_clause_size(8)
    tm.set_lower_ta_threshold(-20)

    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=40, load_best_state=False)
    
    # print("BACKEND:")
    # print(tm._backend_clause_block_cls)
    # print(trainer._cls_feedback_block)
    # print(trainer._cls_dense_ib)
    # print(trainer._cls_sparse_ib)
    # print(trainer._cls_exec_singlethread)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    # seed=6

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


def test_trainer_with_kfold():

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, progress_bar=False, k_folds=20, kfold_progress_bar=True)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    r = trainer.train()

    assert r["best_test_score"] == 1.0


def test_sparse_imdb():


    seed = 42
    rng = np.random.default_rng(seed)  


    imdb = datasets.load_dataset('imdb')
    x, y = imdb['train']['text'], imdb['train']['label']
   
    vectorizer = CountVectorizer(ngram_range=(1, 3), 
                                    binary=True, 
                                    lowercase=True,
                                    max_features=1000,
                                    max_df = 0.8,
                                    min_df = 1,
                                    stop_words='english',
                                    analyzer='word')
    vectorizer.fit(x)


    x_bin = vectorizer.transform(x).toarray().astype(np.uint8)
    y = np.array(y).astype(np.uint32)


    shuffle_index = [i for i in range(len(x))]
    rng.shuffle(shuffle_index)


    x_bin = x_bin[shuffle_index]
    y = y[shuffle_index]


    _train_x_bin, val_x_bin, train_y, val_y = train_test_split(x_bin, y, test_size=0.2, random_state=seed, shuffle=True)


    train_x_bin = csr_matrix(_train_x_bin[:10000])
    train_y = train_y[:10000]
    val_x_bin = csr_matrix(val_x_bin[:2000])
    val_y = val_y[:2000]



    n_clauses = 1000
    s = 2.0
    threshold = 46450
    literal_budget = 7


    n_epochs = 4


    ib = gtc.SparseInputBlock(_train_x_bin.shape[1])
    cb = gtc.ClauseBlockSparse(_train_x_bin.shape[1], n_clauses, len(np.unique(train_y)))
    fb = gtc.FeedbackBlock(len(np.unique(train_y)), threshold, 42)

    ib.set_data(train_x_bin.indices, train_x_bin.indptr, train_y)

    cb.set_feedback(fb)
    cb.set_s(s)
    cb.set_input_block(ib)
    cb.set_active_literals_size(200)
    cb.set_clause_size(100)
    cb.set_lower_ta_threshold(-10)
    cb.initialize()

    exec = gtc.SingleThreadExecutor(ib, [cb], fb, 1, 42)

    best_acc = -1.0
    y_hat = np.empty_like(val_y)

    for epoch in range(3):
        ib.set_data(train_x_bin.indices, train_x_bin.indptr, train_y)
        train_acc = exec.train_epoch()

        ib.set_data(val_x_bin.indices, val_x_bin.indptr, val_y)
        exec.eval_predict(y_hat)

        test_acc = accuracy_score(val_y, y_hat)

        if test_acc > best_acc:
            best_acc = test_acc

        print("Epoch: %d Train: %.3f Test: %.3f" % (epoch+1, train_acc, test_acc))

    print("Best Test Accuracy: %.3f" % best_acc)

    data, indices, indptr = cb.get_clause_state_sparse()
    print(data.shape, indices.shape, indptr.shape)

    # write csr matrix to file
    # df = pd.DataFrame(csr_matrix((data, indices, indptr), shape=(n_clauses*2, _train_x_bin.shape[1])).toarray())
    # df = pd.DataFrame(data)
    # df.to_csv("test.csv", index=False)
        
    # print(csr_matrix((data, indices, indptr), shape=(n_clauses*2, _train_x_bin.shape[1])).toarray())


    # tm = gt.TsetlinMachine(n_literals=train_x_bin.shape[1],
    #                     n_clauses=n_clauses,
    #                     n_classes=len(np.unique(train_y)),
    #                     s=s,
    #                     threshold=threshold,
    #                     literal_budget=literal_budget)


    # tm._backend_clause_block_cls = gtc.ClauseBlockSparse
    # tm.set_active_literals_size(110)
    # # tm.set_clause_size(100)
    # tm.set_lower_ta_threshold(-50)


    # trainer = gt.Trainer(tm=tm,
    #                     n_jobs=6,
    #                     n_epochs=n_epochs,
    #                     seed=seed,
    #                     progress_bar=True,
    #                     load_best_state=False)


    # trainer.set_train_data(csr_matrix(train_x_bin), train_y)
    # trainer.set_test_data(csr_matrix(val_x_bin), val_y)


    # r = trainer.train()
    # print(r)



if __name__ == "__main__":
    # test_trainer_throws_on_wrong_number_of_examples_between_x_and_y()
    # test_train_set_best_state_and_results_afterwards()
    # test_train_simple_xor_py_gtc()
    test_train_simple_xor_sparse()
    test_train_simple_xor()
    # test_train_simple_xor_gtc_tm_backend()
    # test_select_backend_ib()
    # test_set_backend_py_gtc_sparse()

    # test_sparse_imdb()

    # test_trainer_with_kfold()

    print("<done: ", __file__, ">")
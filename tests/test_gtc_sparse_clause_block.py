import numpy as np
import pytest 
import green_tsetlin_core as gtc 
import green_tsetlin as gt
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score


def test_getset_state_and_weights():
    n_literals = 12
    n_clauses = 42 
    n_classes = 3   
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)
    cb.initialize(seed=42)
    
    rng = np.random.default_rng(42)
    offset = 0
    

    
    new_state = rng.integers(low=-50, high=50, size=(n_clauses, n_literals*2)).astype(np.int8)                    
    current_state = np.zeros_like(new_state)
    
    new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses, n_classes)).astype(np.int16)
    current_weights = np.zeros_like(new_weights)
    
    tmp = new_state.copy()
    cb.get_clause_state_sparse()
    assert np.array_equal(tmp, new_state)
    assert np.array_equal(tmp, current_state)
    
    tmp_w = new_weights.copy()
    cb.set_clause_weights(new_weights, offset)
    cb.get_clause_weights(current_weights, offset)
    assert np.array_equal(new_weights, tmp_w)
    assert np.array_equal(current_weights, tmp_w)


def test_simple_xor_sparse():
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42.0
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    # seed=6

    x = csr_matrix(x)
    ex = csr_matrix(ex)
    
    ib = gtc.SparseInputBlock(n_literals)
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)
    ib.set_data(x.indices, x.indptr, y)

    cb.set_s(s)
    fb = gtc.FeedbackBlock(n_classes, threshold, 42)

    
    cb.set_feedback(fb)
    cb.set_input_block(ib)
    cb.initialize()

    exec = gtc.SingleThreadExecutor(ib, [cb], fb, 1, 42)
    
    best_acc = -1.0
    y_hat = np.empty_like(ey)

    for epoch in range(40):
        ib.set_data(x.indices, x.indptr, y)
        train_acc = exec.train_epoch()

        ib.set_data(ex.indices, ex.indptr, ey)
        exec.eval_predict(y_hat)

        test_acc = accuracy_score(ey, y_hat)

        if test_acc > best_acc:
            best_acc = test_acc

        print("Epoch: %d Train: %.3f Test: %.3f" % (epoch+1, train_acc, test_acc))

    print("Best Test Accuracy: %.3f" % best_acc)
    
    data, indices, indptr = cb.get_clause_state_sparse()
    print(data.shape, indices.shape, indptr.shape)
    print(data[-1])
    print(indices[-1])
    print(indptr[-1])
    print(csr_matrix((data, indices, indptr), shape=(n_clauses*2, n_literals)).toarray())


def test_type2_fb_boost_negative_states():
    n_literals = 2

    # ib = gtc.SparseInputBlock(n_literals)

    # x = np.array([[1, 0]], dtype=np.uint32)
    # y = np.array([0], dtype=np.uint32)
    # s = csr_matrix(x)
    
    # ib.set_data(s.indices, s.indptr, y)

    gtc.test_type2_feedback()
    assert False


# def test_set_clause_output():

#     literals = [
#         [1],
#         [],
#         [0, 1],
#         [0]
#     ]

#     for lit in literals:
#         t = gtc.test_train_set_clause_output_sparse(lit)
#         # assert t == [1, 0], t



if __name__ == "__main__":

    # test_getset_state_and_weights()
    # test_set_clause_output()
    # test_type2_fb_boost_negative_states()
    test_simple_xor_sparse()

    print("<done:", __file__, ">")

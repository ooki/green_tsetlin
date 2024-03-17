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
    

    
    new_state = rng.integers(low=-50, high=50, size=(n_clauses*2, n_literals)).astype(np.int8)
    s_new_state = csr_matrix(new_state)
    current_state = np.zeros_like(s_new_state)
    
    new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses, n_classes)).astype(np.int16)
    current_weights = np.zeros_like(new_weights)
    
    tmp = s_new_state.copy()
    cb.set_clause_state_sparse(s_new_state.data, s_new_state.indices, s_new_state.indptr)
    current_state = cb.get_clause_state_sparse()
    assert np.array_equal(tmp.data, s_new_state.data)
    assert np.array_equal(tmp.indices, s_new_state.indices)
    assert np.array_equal(tmp.indptr, s_new_state.indptr)
    assert np.array_equal(tmp.data, current_state[0])
    assert np.array_equal(tmp.indices, current_state[1])
    assert np.array_equal(tmp.indptr, current_state[2]), (tmp.indptr, current_state[2])
    


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
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)

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
    # print(data.shape, indices.shape, indptr.shape)

    print(csr_matrix((data, indices, indptr), shape=(n_clauses*2, n_literals)).toarray())





def test_type2_fb_boost_negative_states():
    n_literals = 2
    n_clauses = 1
    n_classes = 1
    s = 3.0

    x = np.array([[1, 0]], dtype=np.int8)   
    y = np.array([0], dtype=np.int8)
    x = csr_matrix(x)

    ib = gtc.SparseInputBlock(n_literals)
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)

    ib.set_data(x.indices, x.indptr, y)
    fb = gtc.FeedbackBlock(n_classes, 42, 42)
    cb.set_feedback(fb)
    cb.set_input_block(ib)
    cb.set_s(s)
    cb.initialize()

    dense_state = np.array([[-4, -4],
                            [0, 0]])
    
    dense_state = csr_matrix(dense_state)
    # print("DENSE STATES CSR")
    # print(dense_state.data)
    # print(dense_state.indices)
    # print(dense_state.indptr)

    # print("\n DENSE STATES TOARRAY")
    cb.set_clause_state_sparse(dense_state.data.astype(np.int8), dense_state.indices, dense_state.indptr)
    # cb.set_clause_state_sparse(np.array([1, 1, 1, 1, 1, 1], dtype=np.int8), np.array([0, 1, 0, 1, 0, 1], dtype=np.int32), np.array([0, 2, 4, 6], dtype=np.int32))
    # cb.set_clause_weights(np.array([[1, 1], [1, 1]], dtype=np.int16), 0)
    # print('\n')
    for ex in range(1):
        gtc.test_type2_feedback(cb, ib, n_clauses, ex, y[ex], False)

    data, indices, indptr = cb.get_clause_state_sparse()
    # print(data, indices, indptr)
    dense_output = csr_matrix((data, indices, indptr), shape=(n_clauses*2, n_literals)).toarray()
    # print(dense_output)

    expected = np.array([[-4, -3],
                         [0, 0]])

    assert np.array_equal(dense_output, expected), "got: {}, expected: {}".format(dense_output, expected)


def test_type2_AL_fills_clause():
    n_literals = 2
    n_clauses = 2
    n_classes = 1
    s = 3.0

    x = np.array([[0, 1]], dtype=np.int8)   
    y = np.array([0], dtype=np.int8)
    x = csr_matrix(x)

    ib = gtc.SparseInputBlock(n_literals)
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)

    ib.set_data(x.indices, x.indptr, y)
    fb = gtc.FeedbackBlock(n_classes, 42, 42)
    cb.set_feedback(fb)
    cb.set_input_block(ib)
    cb.set_s(s)
    cb.initialize()

    dense_state = np.array([[0, 0],
                            [0, 0],
                            [0, 0],
                            [0, 0]])
    
    dense_state = csr_matrix(dense_state)
    # print("DENSE STATES CSR")
    # print(dense_state.data)
    # print(dense_state.indices)
    # print(dense_state.indptr)

    # print("\n DENSE STATES TOARRAY")
    cb.set_clause_state_sparse(dense_state.data.astype(np.int8), dense_state.indices, dense_state.indptr)
    # cb.set_clause_state_sparse(np.array([1, 1, 1, 1, 1, 1], dtype=np.int8), np.array([0, 1, 0, 1, 0, 1], dtype=np.int32), np.array([0, 2, 4, 6], dtype=np.int32))
    # cb.set_clause_weights(np.array([[1, 1], [1, 1]], dtype=np.int16), 0)
    # print('\n')
    for ex in range(1):
        gtc.test_type2_feedback(cb, ib, n_clauses, ex, y[ex], True)

    data, indices, indptr = cb.get_clause_state_sparse()
    # print(data, indices, indptr)
    dense_output = csr_matrix((data, indices, indptr), shape=(n_clauses*2, n_literals)).toarray()
    # print(dense_output)

    expected = np.array([[-15, 0],
                         [-15, 0],
                         [0, -15],
                         [0, -15]])

    assert np.array_equal(dense_output, expected), "got: {}, expected: {}".format(dense_output, expected)


def test_getset_lower_ta_threshold():
    cb = gtc.ClauseBlockSparse(4, 3, 2)
    # cb.initialize(seed=42)

    first = cb.get_lower_ta_threshold()
    assert first == -20, "got: {}, expected: {}".format(first, -20)

    cb.set_lower_ta_threshold(42)
    second = cb.get_lower_ta_threshold()

    assert second == 42, "got: {}, expected: {}".format(second, 42)

def test_getset_clause_size():
    n_literals = 4
    n_clauses = 3
    n_classes = 2
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)
    # cb.initialize(seed=42)

    first = cb.get_clause_size()
    assert first == n_literals, "got: {}, expected: {}".format(first, n_literals)

    cb.set_clause_size(42)
    second = cb.get_clause_size()
    assert second == 42, "got: {}, expected: {}".format(second, 42)

def test_getset_active_literals_size():
    n_literals = 4
    n_clauses = 3
    n_classes = 2
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)
    # cb.initialize(seed=42)

    first = cb.get_active_literals_size()
    assert first == n_literals, "got: {}, expected: {}".format(first, n_literals)

    cb.set_active_literals_size(42)
    second = cb.get_active_literals_size()

    assert second == 42, "got: {}, expected: {}".format(second, 42)



if __name__ == "__main__":

    # test_simple_xor_sparse()
    
    test_getset_state_and_weights()
    test_type2_fb_boost_negative_states()
    test_type2_AL_fills_clause()
    test_getset_lower_ta_threshold()
    test_getset_clause_size()
    test_getset_active_literals_size()

    print("<done:", __file__, ">")

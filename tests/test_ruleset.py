
import pytest
import numpy as np

from scipy.sparse import csr_matrix

import green_tsetlin as gt


def test_simple_rulset_from_dense():
    dense = gt.DenseState(n_literals=2, n_clauses=3, n_classes=2)
    dense.c = np.array([[1,1,-1,-1],[-1,-1,1,-1],[1,1,-1,-1]], dtype=np.int8)
    dense.w = np.array([[3,-2],[-1,4],[6,-7]], dtype=np.int16)

    rs = gt.ruleset.RuleSet(is_multi_label=False)
    rs.compile_from_dense_state(dense)

    # check that we create rules and weights as lists
    assert isinstance(rs.rules, list)
    assert isinstance(rs.weights, list)

    rules = sorted(rs.rules)
    print(rules)
    assert tuple(rules[0]) == (0, 1)
    assert tuple(rules[1]) == (2,)

    weights = np.array(rs.weights, dtype=int)
    assert np.array_equal(weights[0], [9,-9]) # clause 0 and clause 1 is identical, so they should have been compiled to 1 rule
    assert np.array_equal(weights[1], [-1,4])

    assert rs.n_literals == 2
    assert rs.n_classes == 2


def test_simple_ruleset_from_sparse():
    sparse = gt.SparseState(n_literals=2, n_clauses=3, n_classes=2)
    temp_dense = np.array([[1,1],[-1,-1],[1,1], [-1,-1], [1,0], [-1,-1]], dtype=np.int8)
    temp_sparse = csr_matrix(temp_dense)
    sparse.c_data = [temp_sparse.data]
    sparse.c_indices = [temp_sparse.indices]
    sparse.c_indptr = [temp_sparse.indptr]


    sparse.w = np.array([[3,-2],[-1,4],[6,-7]], dtype=np.int16)
    rs = gt.ruleset.RuleSet(is_multi_label=False)
    rs.compile_from_sparse_state(sparse)

    # check that we create rules and weights as lists
    assert isinstance(rs.rules, list)
    assert isinstance(rs.weights, list)

    rules = sorted(rs.rules)
    print(rules)
    assert tuple(rules[0]) == (0, 1)
    assert tuple(rules[1]) == (2,)

    weights = np.array(rs.weights, dtype=int)
    assert np.array_equal(weights[0], [9,-9]) # clause 0 and clause 1 is identical, so they should have been compiled to 1 rule
    assert np.array_equal(weights[1], [-1,4])

    assert rs.n_literals == 2
    assert rs.n_classes == 2

    sparse = gt.SparseState(n_literals=None, n_clauses=3, n_classes=2)
    temp_dense = np.array([[1,1],[-1,-1],[1,1], [-1,-1], [1,0], [-1,-1]], dtype=np.int8)
    temp_sparse = csr_matrix(temp_dense)
    sparse.c_data = [temp_sparse.data]
    sparse.c_indices = [temp_sparse.indices]
    sparse.c_indptr = [temp_sparse.indptr]

    sparse.w = np.array([[3,-2],[-1,4],[6,-7]], dtype=np.int16)
    rs = gt.ruleset.RuleSet(is_multi_label=False)

    with pytest.warns(UserWarning):
        rs.compile_from_sparse_state(sparse)



if __name__ == "__main__":
    # test_simple_rulset_from_dense()
    test_simple_ruleset_from_sparse()
    print("<done tests:", __file__, ">")


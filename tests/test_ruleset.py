
import pytest
import numpy as np



import green_tsetlin as gt


def test_simple_rulset_from_dense():
    dense = gt.TMState(n_literals=2, n_clauses=3, n_classes=2)
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


if __name__ == "__main__":
    test_simple_rulset_from_dense()
    print("<done tests:", __file__, ">")


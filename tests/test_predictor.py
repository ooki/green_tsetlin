from collections import namedtuple

import pytest
import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc




class MockRuleset:
    def __init__(self):
        self.rules = [[0], [0,1], [0,1], [2,3]]
        self.weights = [[-1, 2], [-3, 4], [5, -6], [7, -8]]
        self.n_literals = 2
        self.n_classes = 2


def test_init():    
    p = gt.Predictor(multi_label=False, explanation="none")
    m = MockRuleset()
    p._set_ruleset(m)    
    p.init()

    assert p.n_literals == 2
    assert p.n_classes == 2


def test_sets_correct_backend_based_on_exploration():
    e_and_backend = [
        ("none", gtc.Inference8u_Ff_Lf_Wf),
        ("literals", gtc.Inference8u_Ff_Lt_Wf),
        # ("features", gtc.Inference8u_Ft_Lf_Wf),
        ("features", None),
        ("positive_weighted_literals", gtc.Inference8u_Ff_Lt_Wt),
        # ("positive_weighted_features", gtc.Inference8u_Ft_Lf_Wt),
        ("positive_weighted_features", None),
    ]

    for explanation, backend_cls in e_and_backend:
        p = gt.Predictor(multi_label=False, explanation=explanation)
        m = MockRuleset()
        p._set_ruleset(m)
        if backend_cls is None:
            with pytest.raises(NotImplementedError):
                p.init()
            continue

        p.init()
        assert p._get_backend() == backend_cls
    

def test_prediction_with_target_names():
    p = gt.Predictor(multi_label=False, explanation="literals")
    p.set_target_names(["a", "b"])
    assert p.target_names == ["a", "b"]
    
    m = MockRuleset()
    p._set_ruleset(m)    
    p.init()

    assert p.predict(np.array([0,0], dtype=np.uint8)) == "a"
    assert p.predict(np.array([1,0], dtype=np.uint8)) == "b"


def test_prediction_literal_explanation():
    p = gt.Predictor(multi_label=False, explanation="literals")
    
    m = MockRuleset()
    p._set_ruleset(m)    
    p.init()

    p.predict(np.array([0,0], dtype=np.uint8))

    x = np.array([1,0], dtype=np.uint8)
    p.predict(x)
    assert np.array_equal(p.explain(explain="target"), [2, 0, 0, 0])
    assert np.array_equal(p.explain(explain="all"), [[-1, 0, 0, 0], [2, 0, 0, 0]])


def test_prediction_positive_clauses_literal_explanation():
    p = gt.Predictor(multi_label=False, explanation="positive_weighted_literals")
    
    m = MockRuleset()
    p._set_ruleset(m)    
    p.init()

    p.predict(np.array([0,0], dtype=np.uint8))

    x = np.array([1,0], dtype=np.uint8)
    p.predict(x)
    assert np.array_equal(p.explain(explain="target"), [2, 0, 0, 0])
    assert np.array_equal(p.explain(explain="all"), [[0, 0, 0, 0], [2, 0, 0, 0]])



if __name__ == "__main__":
    test_init()
    test_sets_correct_backend_based_on_exploration()
    test_prediction_with_target_names()
    test_prediction_literal_explanation()
    test_prediction_positive_clauses_literal_explanation()
    print("<done tests:", __file__, ">")


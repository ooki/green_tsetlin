import numpy as np
import pytest 
import green_tsetlin_core as gtc 




def get_test_rules():
    return [[0], [0,1], [0,1], [2,3]]


def get_test_weights():
    return [[-1, 2], [-3, 4], [5, -6], [7, -8]]

def get_inference_cls(f, l, w):

    if not f and not l and not w:
        inference_cls = gtc.Inference8u_Ff_Lf_Wf
    
    elif not f and l:
        if w:
            inference_cls = gtc.Inference8u_Ff_Lt_Wt
        else:
            inference_cls = gtc.Inference8u_Ff_Lt_Wf
        
    
    assert inference_cls is not None
    

    n_literals = 2
    n_classes = 2
    n_features = 2
    inference = inference_cls(n_literals, n_classes, n_features)


    raw_rules = get_test_rules()
    raw_weights = get_test_weights()
    #features_by_clause = [[0], [1], [0,1], [1]]

    inference.set_rules(raw_rules, raw_weights)
    return n_literals, n_classes, n_features, inference


def test_default_empty_class_is_zero_and_getset_works():

    n_literals, n_classes, n_features, inference = get_inference_cls(False, False, False)
    assert inference.get_empty_class_output() == 0
    inference.set_empty_class_output(1337)
    assert inference.get_empty_class_output() == 1337


def test_predict():
    n_literals, n_classes, n_features, inference = get_inference_cls(False, False, False)

    assert inference.predict(np.array([0,0], dtype=np.uint8)) == 0
    assert inference.predict(np.array([1,0], dtype=np.uint8)) == 1   

def test_predict_take_empty_class_into_account():
    n_literals, n_classes, n_features, inference = get_inference_cls(False, False, False)

    o = inference.predict(np.array([0,1], dtype=np.uint8))
    clauses = inference.get_active_clauses()
    assert clauses.size == 0
    assert inference.get_empty_class_output() == 0
    assert o == 0


    inference.set_empty_class_output(42)
    o = inference.predict(np.array([0,1], dtype=np.uint8))
    clauses = inference.get_active_clauses()
    assert clauses.size == 0
    assert inference.get_empty_class_output() == 42
    assert o == 42


def test_literal_importance():
    
    for w in [(False, -1, 2), (True, 0, 2)]:
        n_literals, n_classes, n_features, inference = get_inference_cls(False, True, w[0])
        inference.predict(np.array([1,0], dtype=np.uint8))     
        
        inference.calculate_explanations(0)                   
        lit_imp0 = inference.get_literal_importance()
        
        inference.calculate_explanations(1)
        lit_imp1 = inference.get_literal_importance()
        
        r = np.zeros(n_literals*2)
        r[0] = lit_imp0[0]
        assert np.array_equal(r, [w[1], 0, 0, 0])
        
        r[0] = lit_imp1[0]
        assert np.array_equal(r, [w[2], 0, 0, 0])


if __name__ == "__main__":
    # test_default_empty_class_is_zero_and_getset_works()
    # test_predict()
    # test_predict_take_empty_class_into_account()
    test_literal_importance()
    print("<done>")
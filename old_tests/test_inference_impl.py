
import pytest
import numpy as np

import green_tsetlin_core as gtc


def get_test_rules():
    return [[0], [0,1], [0,1], [2,3]]

def get_test_inference_object(literal_support):
    n_literals = 2
    n_clauses = 4
    n_classes = 2
    n_features = 2

    if literal_support is True:
        inference = gtc.Inference(n_literals, n_clauses, n_classes, n_features)        
    else:
        inference = gtc.InferenceNoLiteralsImportance(n_literals, n_clauses, n_classes, n_features)


    raw_rules = get_test_rules()
    raw_weights = [[-1, 2], [-3, 4], [5, -6], [7, -8]]
    features_by_clause = [[0], [1], [0,1], [1]]

    inference.set_rules_and_features(raw_rules, raw_weights, features_by_clause)

    return n_literals, n_clauses, n_classes, n_features, inference


def get_test_multi_inference_object():
    n_literals = 2
    n_clauses = 4
    n_category_classes = 3
    n_classes = n_category_classes * 2
    n_features = 2


    inference = gtc.InferenceNoLiteralsImportance(n_literals, n_clauses, n_classes, n_features)

    raw_rules = [[0], [0,1], [0,1], [2,3]]
    raw_weights = [[-1, 2, 1, +1, -2, -1], [-3, 4, 2, +3, -4, -2], [5, -6, -3, -5, +6, +3], [7, -8, 5, -7, +8, +99]] 

    features_by_clause = [[0], [1], [0,1], [1]]

    inference.set_rules_and_features(raw_rules, raw_weights, features_by_clause)

    return n_literals, n_clauses, n_classes, n_features, inference



def test_multi_label_inference():
    n_literals, n_clauses, n_classes, n_features, inference = get_test_multi_inference_object()

    assert np.array_equal(inference.predict_multi(np.array([0,0], dtype=np.uint8)), [1, 0, 0])
    assert np.array_equal(inference.predict_multi(np.array([1,0], dtype=np.uint8)), [0, 1, 1])
    assert np.array_equal(inference.predict_multi(np.array([1,1], dtype=np.uint8)), [1, 1, 1])


def test_default_empty_class_is_zero_and_getset_works():
    n_literals, n_clauses, n_classes, n_features, inference = get_test_inference_object(literal_support=True)
    assert inference.get_empty_class_output() == 0
    inference.set_empty_class_output(1337)
    assert inference.get_empty_class_output() == 1337


def test_get_rules_by_literals():
    n_literals, n_clauses, n_classes, n_features, inference = get_test_inference_object(literal_support=True)
    
    rules = set(tuple(r) for r in get_test_rules())
    
    for k in range(len(rules)):
        rule = inference.get_rule_by_literals(k)
        raw = tuple(rule.tolist())
        assert raw in rules





def test_explantions_return_type_are_numpy_arrays_literal_sup():

    n_literals, n_clauses, n_classes, n_features, inference = get_test_inference_object(literal_support=True)

    x = np.array([0,0], dtype=np.uint8)

    assert isinstance(inference.calc_local_importance(0, False), np.ndarray)    
    assert isinstance(inference.get_cached_literal_importance(), np.ndarray)

    assert isinstance(inference.calculate_global_importance(0, False), np.ndarray)

    assert isinstance(inference.get_active_clauses(), np.ndarray)
    assert inference.get_active_clauses().dtype == np.int32


def test_predict():
    n_literals, n_clauses, n_classes, n_features, inference = get_test_inference_object(literal_support=True)

    assert inference.predict(np.array([0,0], dtype=np.uint8)) == 0
    assert inference.predict(np.array([1,0], dtype=np.uint8)) == 1    
    

def test_predict_take_empty_class_into_account():
    n_literals, n_clauses, n_classes, n_features, inference = get_test_inference_object(literal_support=True)

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


if __name__ == "__main__":
    # test_explantions_return_type_are_numpy_arrays_literal_sup()    
    # test_default_empty_class_is_zero_and_getset_works()
    # test_predict()
    # test_predict_take_empty_class_into_account()
    # test_multi_label_inference()
    tet_get_rules_by_literals()
    print("<done tests:", __file__, ">")






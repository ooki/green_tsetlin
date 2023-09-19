

from collections.abc import Iterable
import random
import math

import pytest
import numpy as np


import green_tsetlin as gt
import green_tsetlin_core as gtc


def test_set_train_test_data_correct_data_format():

    def check_set_train_test_data_throws_valueerror(fn, a, b):
        with pytest.raises(ValueError):
            fn(a,b)
        
    n_examples = 11
    n_literals = 7

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=4, n_classes=2, s=1.5)
    x_right = np.ones(shape=(n_examples, n_literals), dtype=np.uint8)
    x_wrong = np.ones(shape=(n_examples, n_literals), dtype=int)
    
    y_right = np.ones(n_examples, dtype=np.uint32)
    y_wrong = np.ones(n_examples, dtype=int)

    x_wrong_size = np.ones(shape=(n_examples+1, n_literals), dtype=np.uint8)
    y_wrong_size = np.ones(n_examples-1, dtype=np.uint32)

    x_wrong_size_literals = np.ones(shape=(n_examples, n_literals+1), dtype=np.uint8)

    # these should not throw
    tm.set_train_data(x_right)
    tm.set_train_data(x_right, y_right)
    tm.set_test_data(x_right)
    tm.set_test_data(x_right, y_right)

    for fn in [tm.set_train_data, tm.set_test_data]:
        check_set_train_test_data_throws_valueerror(fn, x_wrong, y_wrong)
        check_set_train_test_data_throws_valueerror(fn, x_right, y_wrong)
        check_set_train_test_data_throws_valueerror(fn, x_wrong, y_right) 

        check_set_train_test_data_throws_valueerror(fn, x_wrong_size, y_right)
        check_set_train_test_data_throws_valueerror(fn, x_right, y_wrong_size)

        check_set_train_test_data_throws_valueerror(fn, x_wrong_size_literals, y_right)
    
def test_throws_on_invalid_s():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s_invalid = 0.23
    s_valid = 1.0
    
    gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid)
    with pytest.raises(ValueError):
        gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_invalid)


def test_zero_or_negative_literal_budget_get_set_to_all():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.5
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=-1)
    assert tm.n_literals_budget == n_literals



def test_positive_budget_is_set():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.5
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=1, positive_budget=True)
    assert tm.positive_budget

    
def test_construct_cb_has_correct_properties():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)

    n_blocks = 1
    cbs = tm.construct_clause_blocks(n_blocks=n_blocks)
    assert len(cbs) == n_blocks
    cb = cbs[0]

    assert cb.get_number_of_literals() == n_literals
    assert cb.get_number_of_clauses() == n_clauses
    assert cb.get_number_of_classes() == n_classes
    assert math.isclose(s, cb.get_s()) is True

        
def test_store_state():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)

    n_blocks = 2
    cbs = tm.construct_clause_blocks(n_blocks=n_blocks)    
    
    for cb in cbs:
        cb.initialize()
        cb.set_clause_weight(0, 0, 1337)
        
    cbs[0].set_clause_weight(0, 1, 1338)
    cbs[0].set_clause_weight(0, 2, 1339)
        
    tm.store_state()
    
    print(tm._state["w"].shape)
    #print(tm._state["c"].shape)
    
    print(tm._state["w"].dtype)
    
    print(cbs[0].get_clause_weight(0, 1))
    print(cbs[0].get_clause_weight(0, 2))
    
    tm.set_state(tm._state)
    
    print(cbs[0].get_clause_weight(0, 1))
    print(cbs[0].get_clause_weight(0, 2))

    for cb in cbs:
        cb.cleanup()
        

def test_set_state():
    n_literals = 4
    n_clauses = 7
    n_classes = 3
    s = 2.23


    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)
    tm._tm_cls = gtc.ClauseBlockNV

    n_blocks = 1
    cbs = tm.construct_clause_blocks(n_blocks=n_blocks)    
    cb = cbs[0]
    
    cb.initialize()
          
    #cb.set_clause_weight(0, 1, 1338)        
    #tm.store_state()
    
    n_c = n_clauses * n_literals*2
    n_w = n_clauses * n_classes
    c0 = np.arange(0, n_c).reshape(n_clauses, n_literals*2).astype(dtype=np.int8)
    w0 = np.arange(0, n_w).reshape(n_clauses, n_classes).astype(dtype=np.int16)    
    d0 = {"c": c0, "w": w0}
    tm.set_state(d0)
    
        
    c1 = np.empty_like(c0)
    w1 = np.empty_like(w0)
    c1.fill(13)
    w1.fill(14)
        
    cb.get_clause_state(c1, 0)
    cb.get_clause_weights(w1, 0)
    
    assert np.allclose(c0, c1)
    assert np.allclose(w0, w1)
    
    cb.cleanup()
    
    

def test_get_state_returns_state():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23

    fake_state = "the state"

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)
    tm._state = fake_state

    assert tm.get_state() == fake_state

def test_get_state_throws_if_not_trained():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)

    with pytest.raises(ValueError) as e_info:
        tm.get_state()        
    
    

if __name__ == "__main__":
    # test_set_train_test_data_correct_data_format()
    # test_throws_on_invalid_s()
    # test_zero_or_negative_literal_budget_get_set_to_all()
    # test_construct_cb_has_correct_properties()
    
    # test_store_state()
    
    #test_set_state()
    test_get_state_throws_if_not_trained()
    print("<done tests:", __file__, ">")
    









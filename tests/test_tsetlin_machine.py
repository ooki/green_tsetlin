from collections.abc import Iterable
import random
import math

import pytest
import numpy as np


import green_tsetlin as gt
import green_tsetlin_core as gtc


def test_throws_on_invalid_s():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s_invalid = 0.23
    s_valid = 1.0
    threshold = 10
    
    gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold)
    with pytest.raises(ValueError):
        gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_invalid, threshold=threshold)

def test_throws_on_invalid_threshold():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s_valid = 1.0
    threshold_valid = 10
    threshold_invalid = -2
    
    gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold_valid)
    with pytest.raises(ValueError):
        gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold_invalid)



def test_vanilla_cb_load_state_from_backend():

    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.0
    threshold = 10    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)
    tm._backend_clause_block_cls = gtc.ClauseBlockTM_Lt_Bt
    tm.set_num_clause_blocks(2)
    tm.construct_clause_blocks()

    with gt.allocate_clause_blocks(tm, seed=42):
        tm._load_state_from_backend()
        state0 = tm._state.copy()

        # check that all members of the numpy array state0.c is in the range -2, 2 
        valid_start_states_clauses = [-2, -1, 0, 1, 2]        
        assert np.isin(state0.c, valid_start_states_clauses).all()

        valid_start_states_weights = [-2, -1, 0, 1, 2]
        assert np.isin(state0.w, valid_start_states_weights).all()




    
def test_vanilla_cb_store_state():

    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.0
    threshold = 10    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)
    tm._backend_clause_block_cls = gtc.ClauseBlockTM_Lt_Bt
    tm.set_num_clause_blocks(2)
    tm.construct_clause_blocks()


    with gt.allocate_clause_blocks(tm, seed=42):
        new_weights = np.arange(0, n_classes*n_clauses).astype(np.int16).reshape(n_clauses, n_classes)        
        new_clauses = np.arange(0, n_clauses*(n_literals*2)).astype(np.int8).reshape(n_clauses, n_literals*2)

        tm._state = gt.DenseState()
        tm._state.w = new_weights
        tm._state.c = new_clauses
        state0 = tm._state.copy()
        tm._save_state_in_backend()


        tm._load_state_from_backend()
        state1 = tm._state.copy()

        assert np.array_equal(state1.c, state0.c)
        assert np.array_equal(state1.w, state0.w)



if __name__ == "__main__":
    # test_vanilla_cb_store_state()
    # test_vanilla_cb_load_state_from_backend()

    print("<done:", __file__, ">")
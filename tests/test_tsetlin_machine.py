from collections.abc import Iterable
import random
import math

import pytest
import numpy as np


import green_tsetlin as gt
import green_tsetlin_core as gtc

from scipy.sparse import csr_matrix


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


def test_sparse_cb_load_state_from_backend():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.0
    threshold = 10    
    
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4, boost_true_positives=True, dynamic_AL=True)
    tm.set_num_clause_blocks(2)
    tm.construct_clause_blocks()

    rng = np.random.default_rng(42)
    new_state = rng.integers(low=-50, high=50, size=(n_clauses*2, n_literals)).astype(np.int8)
    s_new_state = csr_matrix(new_state)
    
    new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses, n_classes)).astype(np.int16)
    tmp_w = new_weights.copy()

    new_AL_1 = rng.integers(low=0, high=100, size=(n_classes, n_literals)).astype(np.uint32)
    new_AL_2 = rng.integers(low=0, high=100, size=(n_classes, n_literals)).astype(np.uint32)
    new_AL = [new_AL_1, new_AL_2]
    tmp_AL = new_AL.copy()


    clause_offset = 0
    start = 0
    end = s_new_state.shape[0]//2
    for index, cb in enumerate(tm._cbs):
        cb.initialize(seed=42)
        temp = s_new_state[start:end]
        cb.set_clause_state_sparse(temp.data, temp.indices, temp.indptr)
        cb.set_clause_weights(new_weights, clause_offset)
        cb.set_active_literals(new_AL[index])
        clause_offset += n_clauses//2
        start = end
        end = s_new_state.shape[0]


    tm._load_state_from_backend()
    state0 = tm._state.copy()
    start = 0
    end = s_new_state.shape[0]//2
    for index in range(2):
        assert np.array_equal(state0.c_data[index], s_new_state[start:end].data)
        assert np.array_equal(state0.c_indices[index], s_new_state[start:end].indices)
        assert np.array_equal(state0.c_indptr[index], s_new_state[start:end].indptr)
        assert np.array_equal(state0.w, tmp_w), (state0.w, tmp_w)
        assert np.array_equal(state0.AL[index], tmp_AL[index]), (state0.AL[index], tmp_AL[index])
        
        start = end
        end = s_new_state.shape[0]

def test_sparse_cb_store_state():

    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 1.0
    threshold = 10    
    
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold)
    tm._backend_clause_block_cls = gtc.ClauseBlockTM_Lt_Bt
    tm.set_num_clause_blocks(2)
    tm.construct_clause_blocks()


    with gt.allocate_clause_blocks(tm, seed=42):
        new_weights = np.arange(0, n_classes*n_clauses).astype(np.int16).reshape(n_clauses, n_classes)        

        # should be 2n clauses, but since two blocks do n_clauses per
        new_clauses_1 = csr_matrix(np.arange(0, n_clauses*n_literals).astype(np.int8).reshape(n_clauses, n_literals))
        new_clauses_2 = csr_matrix(np.arange(0, n_clauses*n_literals).astype(np.int8).reshape(n_clauses, n_literals))
        new_clauses_data = [new_clauses_1.data, new_clauses_2.data]
        new_clauses_indices = [new_clauses_1.indices, new_clauses_2.indices]
        new_clauses_indptr = [new_clauses_1.indptr, new_clauses_2.indptr]

        new_AL_1 = np.arange(0, (n_classes)*n_literals).astype(np.uint32).reshape((n_classes), n_literals)
        new_AL_2 = np.arange(0, (n_classes)*n_literals).astype(np.uint32).reshape((n_classes), n_literals)
        new_AL = [new_AL_1, new_AL_2]

        tm._state = gt.SparseState()
        tm._state.w = new_weights
        tm._state.c_data = new_clauses_data
        tm._state.c_indices = new_clauses_indices
        tm._state.c_indptr = new_clauses_indptr
        tm._state.AL = new_AL
        state0 = tm._state.copy()
        tm._save_state_in_backend()


        tm._load_state_from_backend()
        state1 = tm._state.copy()

        assert np.array_equal(state1.c_data, state0.c_data), (state1.c_data, state0.c_data)
        assert np.array_equal(state1.c_indices, state0.c_indices)
        assert np.array_equal(state1.c_indptr, state0.c_indptr)
        assert np.array_equal(state1.AL, state0.AL), (state1.AL, state0.AL)
        assert np.array_equal(state1.w, state0.w)


if __name__ == "__main__":
    # test_vanilla_cb_store_state()
    # test_vanilla_cb_load_state_from_backend()

    # test_sparse_cb_load_state_from_backend()
    test_sparse_cb_store_state()

    print("<done:", __file__, ">")
from collections.abc import Iterable
import random
import math

import pytest
import numpy as np


import green_tsetlin as gt
import green_tsetlin_core as gtc

from scipy.sparse import csr_matrix





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

        print(new_clauses_indptr[0].shape)
        print(new_clauses_indptr[1].shape)

        tm._state = gt.SparseState(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes)
        tm._state.w = new_weights
        tm._state.c_data = new_clauses_data
        tm._state.c_indices = new_clauses_indices
        tm._state.c_indptr = new_clauses_indptr
        tm._state.AL = new_AL
        state0 = tm._state.copy()
        tm._save_state_in_backend()


        tm._load_state_from_backend()
        state1 = tm._state.copy()


        assert np.array_equal(state1.c_data, state0.c_data), (state1.c_data, '\n',state0.c_data)
        assert np.array_equal(state1.c_indices, state0.c_indices), (state1.c_indices, state0.c_indices)
        assert np.array_equal(state1.c_indptr, state0.c_indptr), (state1.c_indptr, state0.c_indptr)
        assert np.array_equal(state1.AL, state0.AL), (state1.AL, state0.AL)
        assert np.array_equal(state1.w, state0.w)


# def test_sparse_state_save_to_file():

#     n_literals = 4
#     n_clauses = 5
#     n_classes = 2
#     s = 3.0
#     threshold = 42
#     tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4, boost_true_positives=True, dynamic_AL=False)        
    
#     tm.active_literals_size = n_literals
#     tm.clause_size = n_literals
#     tm.lower_ta_threshold = -40

#     trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=40, load_best_state=True)
    
#     x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    # seed=6
#     x = csr_matrix(x)
#     ex = csr_matrix(ex)
#     trainer.set_train_data(x, y)
#     trainer.set_test_data(ex, ey)
#     r = trainer.train()
    

#     tm.save_state("/home/tobxtra/data/test_sparse_state_save_to_file.npz")

# def test_sparse_state_load_from_file():

#     state = gt.SparseState(n_literals=4, n_clauses=5, n_classes=2)

#     state = state.load_from_file("/home/tobxtra/data/test_sparse_state_save_to_file.npz")
#     print("w", state.w)
#     print("c_data", state.c_data)
#     print("c_indices", state.c_indices)
#     print("c_indptr", state.c_indptr)
#     print("AL", state.AL)

if __name__ == "__main__":
    # test_vanilla_cb_store_state()
    # test_vanilla_cb_load_state_from_backend()

    # test_sparse_cb_load_state_from_backend()
    test_sparse_cb_store_state()

    # test_sparse_state_save_to_file()
    # test_sparse_state_load_from_file()

    print("<done:", __file__, ">")
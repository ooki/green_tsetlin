import numpy as np
import pytest 
import green_tsetlin_core as gtc 


def test_getset_state_and_weights():
    n_literals = 12
    n_clauses = 42 
    n_classes = 3   
    cb = gtc.ClauseBlockSparse(n_literals, n_clauses, n_classes)
    cb.initialize(seed=42)
    
    rng = np.random.default_rng(42)
    offset = 0
    
    for _ in range(3):
        new_state = rng.integers(low=-50, high=50, size=(n_clauses, n_literals*2)).astype(np.int8)                    
        current_state = np.zeros_like(new_state)
        
        new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses, n_classes)).astype(np.int16)
        current_weights = np.zeros_like(new_weights)
        
        # tmp = new_state.copy()
        # cb.set_clause_state(new_state, offset)
        # cb.get_clause_state(current_state, offset)                
        # assert np.array_equal(tmp, new_state)
        # assert np.array_equal(tmp, current_state)
        
        tmp_w = new_weights.copy()
        cb.set_clause_weights(new_weights, offset)
        cb.get_clause_weights(current_weights, offset)
        assert np.array_equal(new_weights, tmp_w)
        assert np.array_equal(current_weights, tmp_w)




# def test_set_clause_output():

#     literals = [
#         [1],
#         [],
#         [0, 1],
#         [0]
#     ]

#     for lit in literals:
#         t = gtc.test_train_set_clause_output_sparse(lit)
#         # assert t == [1, 0], t


if __name__ == "__main__":

    # test_getset_state_and_weights()
    # test_set_clause_output()

    print("<done:", __file__, ">")

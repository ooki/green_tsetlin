import numpy as np
import pytest 
import green_tsetlin_core as gtc 



def test_getset():
    
    n_literals = 12
    n_clauses = 42 
    n_classes = 3   
    cb = gtc.ClauseBlockTM_Lt_Bt(n_literals, n_clauses, n_classes)
    
    assert n_literals == cb.get_number_of_literals()
    assert n_clauses == cb.get_number_of_clauses()
    assert n_classes == cb.get_number_of_classes()
    
    cb.set_s(5.0)
    assert abs(cb.get_s() - 5.0) < 1e-7
    
    cb.set_s(9.0)
    assert abs(cb.get_s() - 9.0) < 1e-7
    
    cb.set_literal_budget(42)
    assert cb.get_literal_budget() == 42
    
    cb.set_literal_budget(3)
    assert cb.get_literal_budget() == 3
    
    
def test_initialize_cleanup_flags():
    n_literals = 12
    n_clauses = 42 
    n_classes = 3   
    cb = gtc.ClauseBlockTM_Lt_Bt(n_literals, n_clauses, n_classes)
    
    assert cb.is_initialized() is False
    cb.initialize(seed=42)
    assert cb.is_initialized() is True
    
    cb.cleanup()
    assert cb.is_initialized() is False
    
    
def test_getset_state_and_weights():
    n_literals = 12
    n_clauses = 42 
    n_classes = 3   
    cb = gtc.ClauseBlockTM_Lt_Bt(n_literals, n_clauses, n_classes)
    cb.initialize(seed=42)
    
    rng = np.random.default_rng(42)
    offset = 0
    
    for _ in range(3):
        new_state = rng.integers(low=-50, high=50, size=(n_clauses, n_literals*2)).astype(np.int8)                    
        current_state = np.zeros_like(new_state)
        
        new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses, n_classes)).astype(np.int16)
        current_weights = np.zeros_like(new_weights)
        
        tmp = new_state.copy()
        cb.set_clause_state(new_state, offset)
        cb.get_clause_state(current_state, offset)                
        assert np.array_equal(tmp, new_state)
        assert np.array_equal(tmp, current_state)
        
        tmp_w = new_weights.copy()
        cb.set_clause_weights(new_weights, offset)
        cb.get_clause_weights(current_weights, offset)
        assert np.array_equal(new_weights, tmp_w)
        assert np.array_equal(current_weights, tmp_w)
        
        
    offset = 100
    for _ in range(3):
        after = n_clauses+offset
        new_state = rng.integers(low=-50, high=50, size=(n_clauses+offset+10, n_literals*2)).astype(np.int8)                    
        new_state[0:offset, :] = -42
        new_state[after, :] = 42
        current_state = np.zeros_like(new_state)        
                
        new_weights = rng.integers(low=-1000, high=1000, size=(n_clauses+offset+10, n_classes)).astype(np.int16)
        new_weights[0:offset, :] = -69
        new_weights[after, :] = 69
        current_weights = np.zeros_like(new_weights)
        
        cb.set_clause_state(new_state, offset)
        cb.get_clause_state(current_state, offset)            
        assert np.array_equal(new_state[offset:after], current_state[offset:after])
        assert (current_state[0:offset] == 0).all() # make sure we dont overwrite anything - pre
        assert (current_state[after] == 0).all() #  after
        
        cb.set_clause_weights(new_weights, offset)
        cb.get_clause_weights(current_weights, offset)
        assert np.array_equal(new_weights[offset:after], current_weights[offset:after])
        assert (current_weights[0:offset] == 0).all() # make sure we dont overwrite anything - pre
        assert (current_weights[after] == 0).all() #  after

    cb.cleanup()
    
    
if __name__ == "__main__":
    test_getset()
    test_initialize_cleanup_flags()
    test_getset_state_and_weights()
    
    print("<done:", __file__, ">")
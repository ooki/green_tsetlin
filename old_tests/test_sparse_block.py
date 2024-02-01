
import pytest
pytest.skip(allow_module_level=True)



import numpy as np
import green_tsetlin_core as gtc

def test_can_init():
    n_literals = 2
    n_classes = 2
    n_clauses = 4

    cb = gtc.SparseClauseBlockNV(n_literals, n_clauses, n_classes, 1 , 1)
    cb.initialize(42)
    
    #c_state = np.array(cb.get_copy_clause_states(), dtype=np.int8).reshape(n_clauses, -1)    
    #cb.cleanup()

    #assert np.isin(c_state[:, 0:2], [-1, 0]).all()    
    #assert np.isin(c_state[:, 2:4], [-1, 0]).all()


if __name__ == "__main__":
    test_can_init()    
    print("<done tests:", __file__, ">")
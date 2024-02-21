import numpy as np
import pytest 
import green_tsetlin_core as gtc 



def test_cannot_construct_base_clause_block():
    
    with pytest.raises(TypeError):
        gtc.ClauseBlock()
    
    
if __name__ == "__main__":
    test_cannot_construct_base_clause_block()
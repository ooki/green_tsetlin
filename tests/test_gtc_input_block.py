
import numpy as np
import pytest 
import green_tsetlin_core as gtc 


def test_DenseInputBlock_check_multi_flag():
    
    n_examples = 10
    n_literals = 4

    ib_single = gtc.DenseInputBlock(n_literals)    
    x = np.zeros((n_examples, n_literals), dtype=np.uint8)
    y = np.zeros(n_examples, dtype=np.uint32)    
    ib_single.set_data(x, y)
    assert ib_single.is_multi_label() == False

    ib_multi = gtc.DenseInputBlock(n_literals)
    y_multi = np.zeros(shape=(n_examples, 5), dtype=np.uint32)    
    ib_multi.set_data(x, y_multi)
    assert ib_multi.is_multi_label() == True


def test_throws_on_wrong_size_arrays():
    
    n_examples = 10
    n_literals = 4

    ib_single = gtc.DenseInputBlock(n_literals)    
    x = np.zeros((n_examples, n_literals), dtype=np.uint8)
    x_half = np.zeros((n_examples//2, n_literals), dtype=np.uint8)
    
    y = np.zeros(n_examples, dtype=np.uint32)    
    y_half = np.zeros(n_examples//2, dtype=np.uint32)    
    
    with pytest.raises(RuntimeError):
        ib_single.set_data(x, y_half)
        
    with pytest.raises(RuntimeError):
        ib_single.set_data(x_half, y)
    

if __name__ == "__main__":
    test_DenseInputBlock_check_multi_flag()
    test_throws_on_wrong_size_arrays()


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
    

def test_SparseInputDenseOutputBlock_throws_on_dense():
    n_examples = 10
    n_literals = 4

    ib_sido = gtc.SparseInputDenseOutputBlock(n_literals)    
    x = np.zeros((n_examples, n_literals), dtype=np.uint8)    
    y = np.zeros(n_examples, dtype=np.uint32)    
    
    with pytest.raises(TypeError):
        ib_sido.set_data(x, y)


def test_im2col_give_correct_shapes():

    n_examples = 1
    width = 4
    height = 4
    channels = 1
    image = np.arange(0, 16).reshape(n_examples, width, height, channels)
    image = image.astype(np.uint8)

    col = gtc.im2col(image, 2, 2)
    assert np.array_equal(col.shape, [1, 9, 8])

    col = gtc.im2col(image, 3, 3)
    assert np.array_equal(col.shape, [1, 4, 11])

    col = gtc.im2col(image, 3, 4)
    assert np.array_equal(col.shape, [1, 2, 13])

    col = gtc.im2col(image, 4, 4)
    assert np.array_equal(col.shape, [1, 1, 16])

if __name__ == "__main__":
    # test_DenseInputBlock_check_multi_flag()
    # test_throws_on_wrong_size_arrays()
    # test_im2col_give_correct_shapes()
    test_SparseInputDenseOutputBlock_throws_on_dense()

    print("<done tests:", __file__, ">")

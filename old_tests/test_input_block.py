
import numpy as np
import green_tsetlin_core as gtc


def test_DenseInputBlock_set_as_label_block_no_multi():
    
    n_examples = 10
    n_literals = 4
    
    ib_x = gtc.DenseInputBlock(n_literals)
    ib_xy = gtc.DenseInputBlock(n_literals)
    
    x0 = np.zeros((n_examples, n_literals), dtype=np.uint8)
    x1 = np.ones((n_examples, n_literals), dtype=np.uint8)
    y = np.zeros(n_examples, dtype=np.int32)
    
    y_empty = np.empty(shape=(0,0), dtype=np.int32)
    
    ib_xy.set_data(x1, y)
    ib_x.set_data(x0, y_empty)
    
    assert ib_xy.is_label_block() == True
    assert ib_xy.is_multi_label() == False
    assert ib_xy.get_num_labels_per_example() == 1
    assert ib_xy.get_number_of_examples() == n_examples


    assert ib_x.is_label_block() == False
    assert ib_x.is_multi_label() == False


def test_DenseInputBlock_set_as_label_block_no_multi_2d_labels():
    
    n_examples = 10
    n_literals = 4
    n_classes = 1

    
    ib_xy = gtc.DenseInputBlock(n_literals)
    
    x = np.zeros((n_examples, n_literals), dtype=np.uint8)
    y = np.ones((n_examples, n_classes), dtype=np.int32)
    
    ib_xy.set_data(x, y)
        
    assert ib_xy.is_multi_label() == False
    assert ib_xy.get_num_labels_per_example() == 1

def test_DenseInputBlock_set_as_label_block_multi():
    
    n_examples = 10
    n_literals = 4
    n_classes = 5
    
    ib_x = gtc.DenseInputBlock(n_literals)
    ib_xy = gtc.DenseInputBlock(n_literals)
    
    x0 = np.zeros((n_examples, n_literals), dtype=np.uint8)
    x1 = np.ones((n_examples, n_literals), dtype=np.uint8)
    y = np.ones((n_examples, n_classes), dtype=np.int32)
    
    y_empty = np.empty(shape=(0,0), dtype=np.int32)
    
    ib_xy.set_data(x1, y)
    ib_x.set_data(x0, y_empty)
    
    assert ib_xy.is_label_block() == True
    assert ib_xy.is_multi_label() == True
    assert ib_xy.get_num_labels_per_example() == n_classes

    assert ib_x.is_label_block() == False
    assert ib_x.is_multi_label() == False

    
    
if __name__ == "__main__":
    test_DenseInputBlock_set_as_label_block_no_multi()    
    test_DenseInputBlock_set_as_label_block_multi()
    test_DenseInputBlock_set_as_label_block_no_multi_2d_labels()
    print("<done tests:", __file__, ">")

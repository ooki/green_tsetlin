from collections import namedtuple

import pytest

import numpy as np

import green_tsetlin as gt



def test_checks_for_inconsistent_number_of_classes():
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s = 2.23
    threshold = 42
    tm0 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)
    tm1 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes+1, s=s)

    trainer = gt.Trainer(threshold=threshold, n_epochs=0)
    with pytest.raises(ValueError):
        trainer.train([tm0, tm1])

    x = np.ones([2, n_literals], dtype=np.uint8)
    y = np.ones([2], dtype=np.uint32)

    tm0 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)        
    tm1 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s)
    
    tm0.set_train_data(x, y)
    tm1.set_train_data(x)
    trainer.train([tm0, tm1])
    

def test_checks_threshold_param():
    threshold_negative = -42
    threshold_good = 42

    gt.Trainer(threshold=threshold_good)
    with pytest.raises(ValueError):
        gt.Trainer(threshold=threshold_negative)



def test_train_xor_no_crash():

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42
    tm0 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=4)
    tm1 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=4)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)
    tm0.set_train_data(x, y)
    tm0.set_test_data(ex, ey)
    tm1.set_train_data(x)
    tm1.set_test_data(ex)

    trainer = gt.Trainer(threshold, seed=32, n_jobs=20)
    trainer.train([tm0, tm1])
    
    assert tm0._state is not None
    
def test_train_epoch_callback_gets_called():
    
    counter = [0]
    def epoch_cb(a,b,c):
        counter[0] += 1
        
    
    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42
    tm0 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=4)
    tm1 = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=4)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)
    tm0.set_train_data(x, y)
    tm0.set_test_data(ex, ey)
    tm1.set_train_data(x)
    tm1.set_test_data(ex)

    trainer = gt.Trainer(threshold, n_epochs=2, seed=32, n_jobs=20, early_exit_acc=False, fn_epoch_callback=epoch_cb)
    trainer.train([tm0, tm1])
    
    assert tm0._state is not None
    assert counter[0] > 0


def test_allocate_correct_number_of_blocks():
    n_literals = 3
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=4)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)
    tm.set_train_data(x, y)
    tm.set_test_data(ex, ey)

    n_jobs = 21
    trainer = gt.Trainer(threshold, n_epochs=3, seed=32, n_jobs=n_jobs, early_exit_acc=True)
    trainer.train(tm)

    assert trainer.n_blocks_used == n_jobs



def test_block_allocation_works_for_multiple_and_single_jobs():

    tm_nt = namedtuple("TsetlinMachine", "n_clauses")
    def get_tms(cs):
        return [tm_nt(c) for c in cs]
    
    # 1 job -> multiple tms : all gets 1 block
    trainer0 = gt.Trainer(1, n_epochs=2, seed=32, n_jobs=1)    
    blocks0 = trainer0._calculate_blocks_per_tm(get_tms([50]*5))
    assert len(blocks0) == 5
    assert all([b == 1 for b in blocks0])
    
    
    # 6 jobs, should allocate as 1, 2, 3
    trainer1 = gt.Trainer(1, n_epochs=2, seed=32, n_jobs=6)
    blocks1 = trainer1._calculate_blocks_per_tm(get_tms([5, 30, 100])) 
    # => counts [5, 30->15, 100->50->25]   
    assert blocks1 == [1, 2, 3]


    # 6 job -> 1 tms : gets 6 blocks
    trainer2 = gt.Trainer(1, n_epochs=2, seed=32, n_jobs=6)    
    blocks2 = trainer2._calculate_blocks_per_tm(get_tms([100]))
    assert len(blocks2) == 1
    assert blocks2[0] == 6


if __name__ == "__main__":
#     test_checks_for_inconsistent_number_of_classes()
#     test_checks_threshold_param()
#     test_train_xor_no_crash()
#     test_train_epoch_callback_gets_called()
    
    test_allocate_correct_number_of_blocks()

    print("<done tests:", __file__, ">")
    


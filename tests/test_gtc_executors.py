
import numpy as np
import pytest 
import green_tsetlin_core as gtc 



def test_single_exec_throws_if_threads_is_not_1():
    n_literals = 8
    n_classes = 2
    n_clauses = 2
    threshold = 10
    seed = 42
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockTM_Lt_Bt(n_literals, n_clauses, n_classes)
    cb.initialize(seed)
        
    feedback_block = gtc.FeedbackBlock(n_classes, threshold, seed)
    cb.set_feedback(feedback_block)     
    cb.set_input_block(ib)
    
    n_threads_correct = 1
    n_threads_incorrect = 10
    exec = gtc.SingleThreadExecutor(ib, [cb], feedback_block, n_threads_correct, seed)
    with pytest.raises(RuntimeError):
        exec = gtc.SingleThreadExecutor(ib, [cb], feedback_block, n_threads_incorrect, seed)


if __name__ == "__main__":
    test_single_exec_throws_if_threads_is_not_1()

import pytest
import numpy as np

import green_tsetlin_core as gtc
import green_tsetlin as gt


# only run if we actually can import Neon Block
try:
    cls = gtc.ClauseBlockNeon
except AttributeError:
    pytest.skip(allow_module_level=True)


def test_ClauseBlockNeon_allocate_cleanup():
    cb = gtc.ClauseBlockNeon(4, 3, 2)
    cb.initialize()
    cb.cleanup()

def test_ClauseBlockNeon_allocate_extra_mem_to_align():

    n_classes = 2

    def check_allocated_mem(n_literals, n_clauses, c_size):
        cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
        cb.initialize()
        c_state = np.array(cb.get_copy_clause_states(), dtype=np.int8).reshape(n_clauses, -1)
        assert c_state.shape == c_size
        cb.cleanup()

    # neon use 16 ta's per vector 
    check_allocated_mem(n_literals=2, n_clauses=2, c_size=(2, 32))
    check_allocated_mem(n_literals=31, n_clauses=5, c_size=(5, 64))
    check_allocated_mem(n_literals=71, n_clauses=3, c_size=(3, 160))



def test_ClauseBlockNEON_train_set_clause_output_and_set_votes():
    n_literals = 2
    n_classes = 2
    n_clauses = 6
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    train_x = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    train_y = np.array([1, 0], dtype=np.uint32)
    
    ib.set_data(train_x, train_y)
    #print("example:", train_x[0], "neg:", (train_x[0]+1)%2)

    for c in range(n_clauses):
        for k in range(n_literals):        
            cb.set_ta_state(c, k, True, -10)
            cb.set_ta_state(c, k, False, -10)


    # set clauses
    cb.set_ta_state(0, 0, True, 7)
    cb.set_ta_state(0, 1, False, 7)
    
    cb.set_ta_state(1, 1, True, 7)
    cb.set_ta_state(1, 0, False, 7)

    cb.set_ta_state(3, 0, False, 7)

    cb.set_ta_state(4, 1, True, 7)

    cb.set_ta_state(5, n_literals+1, True, 9)
    cb.set_ta_state(5, n_literals+2, True, 9)

    
    ib.prepare_example(0)
    gtc.time_train_set_clause_output_and_set_votes(cb)

    out = cb.get_copy_clause_outputs()
    # print("out:", out)

    
    assert out[0] == 1 # correct ta state
    assert out[1] == 0 # pos and neg wrong wrt. input
    assert out[2] == 1 # empty clause
    assert out[3] == 0 # neg wrong
    assert out[4] == 0 # pos wrong
    assert out[5] == 1 # empty clause : IF mask is used correctly
    
    cb.cleanup()
    
def test_ClauseBlockNEON_train_set_clause_output_and_set_votes_chunk0():
    n_literals = 2+16
    n_classes = 2
    n_clauses = 5
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    train_x = np.zeros(shape=(2, n_literals), dtype=np.uint8)
    train_x[0, 0] = 1
    train_x[0, 1] = 0

    train_y = np.array([1, 0], dtype=np.uint32)
    
    ib.set_data(train_x, train_y)
    #print("example:", train_x[0], "neg:", (train_x[0]+1)%2)

    for c in range(n_clauses):
        for k in range(n_literals):        
            cb.set_ta_state(c, k, True, -10)
            cb.set_ta_state(c, k, False, -10)


    # set clauses
    cb.set_ta_state(0, 0, True, 7)
    cb.set_ta_state(0, 1, False, 7)
    
    cb.set_ta_state(1, 1, True, 7)
    cb.set_ta_state(1, 0, False, 7)

    cb.set_ta_state(3, 0, False, 7)

    cb.set_ta_state(4, 1, True, 7)
    
    ib.prepare_example(0)
    gtc.time_train_set_clause_output_and_set_votes(cb)

    out = cb.get_copy_clause_outputs()
    #print("out:", out)

    
    assert out[0] == 1 # correct ta state
    assert out[1] == 0 # pos and neg wrong wrt. input
    assert out[2] == 1 # empty clause
    assert out[3] == 0 # neg wrong
    assert out[4] == 0 # pos wrong
    
    cb.cleanup()
    

def test_ClauseBlockNEON_eval_set_clause_output_and_set_votes_empty_clause_is_off():
    n_literals = 2
    n_classes = 2
    n_clauses = 10
    threshold = 100
    s_param = 5.0

        
    ib = gtc.DenseInputBlock(n_literals)
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    train_x = np.array([[1, 0]], dtype=np.uint8)
    train_y = np.array([1], dtype=np.uint32)
    
    ib.set_data(train_x, train_y)

    for c in range(n_clauses):
        for k in range(n_literals):        
            cb.set_ta_state(c, k, True, -10)
            cb.set_ta_state(c, k, False, -10)

    ib.prepare_example(0)    
    gtc.time_eval_set_clause_output_and_set_votes(cb)
    out = cb.get_copy_clause_outputs()

    
    assert out[0] == 0 # empty is off since we are in eval
    cb.cleanup()


def test_SetClauseOutputNeon_literal_counts_reminder():
    n_literals = 4
    n_classes = 1
    n_clauses = 6
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)

    neon_ta_per_chunk = 16
    for c in range(n_clauses):
        for k in range(neon_ta_per_chunk):        
            cb.set_ta_state(c, k, True, -10)
            cb.set_ta_state(c, k, False, -10)
        
    cb.set_ta_state(0, 2, True, 3)    
    cb.set_ta_state(0, 3, True, 4)

    cb.set_ta_state(1, 0, False, 5)

    cb.set_ta_state(2, 2, True, 3)    
    cb.set_ta_state(2, 3, True, 4)
    cb.set_ta_state(2, 0, False, 5)

    # 3 is empty

    cb.set_ta_state(4, 2, True, 3)    
    cb.set_ta_state(4, 3, True, 4)

    # make sure we correctly mask the input
    cb.set_ta_state(5, 2, True, 3)
    cb.set_ta_state(5, 14, True, 4) # these  2 are outside the bounds and should not affect the count
    cb.set_ta_state(5, 14, False, 5)
            
    ib.prepare_example(0)
    gtc.time_train_set_clause_output_and_set_votes(cb)

    counts = cb.get_copy_literal_counts()
    assert counts[0] == 2 # positive
    assert counts[1] == 1 # negated
    assert counts[2] == 3 # pos+neg
    assert counts[3] == 0 # empty
    assert counts[4] > 0  # clause_output = 0 -> dont reset the count
    assert counts[5] == 1 # make sure the masked ta's dont affect the count

    cb.cleanup()
    

def test_xor_train_to_100_neon():
    n_literals = 10
    n_clauses = 16
    n_classes = 2
    s = 3.0
    n_literal_budget = 2
    threshold = 15    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=n_literal_budget)
    assert tm._tm_cls == gtc.ClauseBlockNeon

    x, y, ex, ey = gt.dataset_generator.xor_dataset(noise=0.25, n_literals=n_literals, n_train=5000, n_test=500, seed=41)
    tm.set_train_data(x, y)
    tm.set_test_data(ex, ey)

    best_test_acc = 0.0
    for i in range(0, 5):
        trainer = gt.Trainer(threshold, n_epochs=100, seed=32+i, n_jobs=0, early_exit_acc=True, progress_bar=False)
        r = trainer.train(tm)
        best_test_acc = max(best_test_acc, r["best_test_score"])

        if r["did_early_exit"]:            
            break

    assert best_test_acc > 0.95


def test_Type2_feedback_ignore_positive_states():
    n_literals = 1
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[1]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 0, True, 17)    
    cb.set_ta_state(0, 0, False, 18)
    
    clause_index = 0
    for _ in range(100):
        gtc.time_type2_feedback_neon(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 0, True) == 17
    assert cb.get_ta_state(0, 0, False) == 18
    cb.cleanup()


def test_Type2_feedback_inc_negative_states_with_0_literal():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[1, 0]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 0, True, -14)
    cb.set_ta_state(0, 0, False, -15)
    cb.set_ta_state(0, 1, True, -26)    
    cb.set_ta_state(0, 1, False, -27)
    
    clause_index = 0
    for _ in range(1):
        gtc.time_type2_feedback_neon(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 0, True) == -14       # lit:1
    assert cb.get_ta_state(0, 0, False) == -15+1    # lit:0
    assert cb.get_ta_state(0, 1, True) == -26+1     # lit:0
    assert cb.get_ta_state(0, 1, False) == -27      # lit:1
    
    cb.cleanup()
    
    
def test_Type2_feedback_inc_negative_states_with_0_literal_stop_at_zero():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[1, 0]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 0, True, -17)
    cb.set_ta_state(0, 0, False, -18)
    cb.set_ta_state(0, 1, True, -22)    
    cb.set_ta_state(0, 1, False, 23)
    
    clause_index = 0
    for _ in range(50):
        gtc.time_type2_feedback_neon(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 0, True) == -17       # lit:1
    assert cb.get_ta_state(0, 0, False) == 0    # lit:0
    assert cb.get_ta_state(0, 1, True) == 0     # lit:0
    assert cb.get_ta_state(0, 1, False) == 23      # lit:1
    
    cb.cleanup()


def test_Type2_feedback_inc_negative_states_with_0_literal_stop_at_zero_next_chunk():
    n_literals = 16+2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.zeros(16+2, dtype=np.uint8).reshape(1, -1)
    x[0, 16] = 1
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 16, True, -17)
    cb.set_ta_state(0, 16, False, -18)
    cb.set_ta_state(0, 17, True, -22)    
    cb.set_ta_state(0, 17, False, 23)
    
    clause_index = 0
    for _ in range(50):
        gtc.time_type2_feedback_neon(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 16, True) == -17      # lit:1
    assert cb.get_ta_state(0, 16, False) == 0       # lit:0
    assert cb.get_ta_state(0, 17, True) == 0        # lit:0
    assert cb.get_ta_state(0, 17, False) == 23      # lit:1

    cb.cleanup()


def test_Type1a_feedback_increase_states():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 2.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[1, 0]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 0, True, 1)    
    cb.set_ta_state(0, 0, False, -2)

    cb.set_ta_state(0, 1, True, -3)
    cb.set_ta_state(0, 1, False, 4)

    clause_index = 0
    for _ in range(1500):
        gtc.time_type1a_feedback_neon(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 0, True) >= 127
    assert cb.get_ta_state(0, 0, False) <= -127
    assert cb.get_ta_state(0, 1, True) <= -127
    assert cb.get_ta_state(0, 1, False) >= 127

    cb.cleanup()


def test_Type1b_feedback_increase_states():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 1.05
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNeon(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize(42)
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[1, 0]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)
    
    cb.set_ta_state(0, 0, True, 20)
    cb.set_ta_state(0, 0, False, -20)

    cb.set_ta_state(0, 1, True, 30)
    cb.set_ta_state(0, 1, False, -30)

    state = np.zeros((1, n_literals*2), dtype=np.int8)
    cb.get_clause_state(state, 0)
    
    clause_index = 0
    for _ in range(1500):
        gtc.time_type1b_feedback_neon(cb, ib, clause_index)
    
    cb.get_clause_state(state, 0)

    assert cb.get_ta_state(0, 0, True) <= -127
    assert cb.get_ta_state(0, 0, False) <= -127
    assert cb.get_ta_state(0, 1, True) <= -127
    assert cb.get_ta_state(0, 1, False) <= -127


    cb.cleanup()


# def test_xor_train_to_100_neon():
#     n_literals = 10
#     n_clauses = 16
#     n_classes = 2
#     s = 3.0
#     n_literal_budget = 2
#     threshold = 15    
    
#     tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=n_literal_budget)
#     assert tm._tm_cls == gtc.ClauseBlockNeon

#     x, y, ex, ey = gt.dataset_generator.xor_dataset(noise=0.25, n_literals=n_literals, n_train=5000, n_test=500, seed=41)
#     tm.set_train_data(x, y)
#     tm.set_test_data(ex, ey)

#     best_test_acc = 0.0
#     for i in range(0, 5):
#         trainer = gt.Trainer(threshold, n_epochs=100, seed=32+i, n_jobs=0, early_exit_acc=True, progress_bar=False)
#         r = trainer.train(tm)
#         best_test_acc = max(best_test_acc, r["best_test_score"])

#         print("best acc:", best_test_acc)
#         if r["did_early_exit"]:            
#             break

#     assert best_test_acc > 0.95

if __name__ == "__main__":

    # test_ClauseBlockNeon_allocate_cleanup()
    # test_ClauseBlockNeon_allocate_extra_mem_to_align()

    #test_ClauseBlockNEON_train_set_clause_output_and_set_votes()
    #test_ClauseBlockNEON_train_set_clause_output_and_set_votes_chunk0()

    #test_ClauseBlockNEON_eval_set_clause_output_and_set_votes_empty_clause_is_off()

    #test_Type2_feedback_inc_negative_states_with_0_literal_stop_at_zero()
    #test_Type2_feedback_inc_negative_states_with_0_literal_stop_at_zero_next_chunk()
    #test_Type2_feedback_inc_negative_states_with_0_literal()

    # test_Type1a_feedback_increase_states()
    #test_tmp_debug()
    #gtc.test_neon()

    test_xor_train_to_100_neon()
    print("<done tests:", __file__, ">")

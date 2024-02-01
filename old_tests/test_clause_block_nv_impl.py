
import numpy as np
import green_tsetlin_core as gtc
import green_tsetlin as gt


def test_ClauseBlockNV_init_clauses():
    n_literals = 2
    n_classes = 2
    n_clauses = 4

    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    cb.initialize()
    
    c_state = np.array(cb.get_copy_clause_states(), dtype=np.int8).reshape(n_clauses, -1)    
    cb.cleanup()

    assert np.isin(c_state[:, 0:2], [-1, 0]).all()    
    assert np.isin(c_state[:, 2:4], [-1, 0]).all()

    
def test_ClauseBlockNV_train_set_clause_output_and_set_votes():
    n_literals = 2
    n_classes = 2
    n_clauses = 5
    threshold = 100
    s_param = 5.0
            
    ib = gtc.DenseInputBlock(n_literals)
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    train_x = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    train_y = np.array([1, 0], dtype=np.uint32)
    
    ib.set_data(train_x, train_y)
    # print("example:", train_x[0], "neg:", (train_x[0]+1)%2)

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

    
    assert out[0] == 1 # correct ta state
    assert out[1] == 0 # pos and neg wrong wrt. input
    assert out[2] == 1 # empty clause
    assert out[3] == 0 # neg wrong
    assert out[4] == 0 # pos wrong
    
    cb.cleanup()
    

def test_Type1a_feedback_increase_states():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 2.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize(42)
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


    # state = np.zeros((1, n_literals*2), dtype=np.int8)
    # cb.get_clause_state(state, 0)

    # print("state:")
    # print(state)
    
    clause_index = 0
    for _ in range(1500):
        gtc.time_type1a_feedback_nv(cb, ib, clause_index)
    
    #assert cb.get_ta_state(0, 0, True) == 17
    #assert cb.get_ta_state(0, 0, False) == 18

    assert cb.get_ta_state(0, 0, True) >= 127
    assert cb.get_ta_state(0, 0, False) <= -127
    assert cb.get_ta_state(0, 1, True) <= -127
    assert cb.get_ta_state(0, 1, False) >= 127

    # cb.get_clause_state(state, 0)

    # print("state:")
    # print(state)


    cb.cleanup()


def test_Type1b_feedback_increase_states():
    n_literals = 2
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 2.5
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
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

    print("state0:")
    print(state)
    
    clause_index = 0
    for _ in range(1500):
        gtc.time_type1b_feedback_nv(cb, ib, clause_index)
    
    cb.get_clause_state(state, 0)

    print("state 1:")
    print(state)

    assert cb.get_ta_state(0, 0, True) <= -127
    assert cb.get_ta_state(0, 0, False) <= -127
    assert cb.get_ta_state(0, 1, True) <= -127
    assert cb.get_ta_state(0, 1, False) <= -127

    cb.cleanup()


    
def test_Type2_feedback_ignore_positive_states():
    n_literals = 1
    n_classes = 1
    n_clauses = 1
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
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
        gtc.time_type2_feedback_nv(cb, ib, clause_index)
    
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
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
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
        gtc.time_type2_feedback_nv(cb, ib, clause_index)
    
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
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
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
        gtc.time_type2_feedback_nv(cb, ib, clause_index)
    
    assert cb.get_ta_state(0, 0, True) == -17       # lit:1
    assert cb.get_ta_state(0, 0, False) == 0    # lit:0
    assert cb.get_ta_state(0, 1, True) == 0     # lit:0
    assert cb.get_ta_state(0, 1, False) == 23      # lit:1
    
    cb.cleanup()



def test_CoaleasedTsetlinStateNV_get_clause_state():
    
    n_literals = 4
    n_classes = 2
    n_clauses = 3

    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    cb.initialize()
    dst = np.empty(shape=(n_clauses+2, n_literals*2), dtype=np.int8)
    dst.fill(-69)    

    state = np.array([[-1, -2, -3, -4, 1, 2, 3, 4],
                     [-10, -20, -30, -40, 10, 20, 30, 40],
                     [-11, -22, -33, -44, 11, 22, 33, 44]], dtype=np.int8)
    
    for clause_k in range(n_clauses):
        for lit_i in range(n_literals):
            cb.set_ta_state(clause_k, lit_i, True, state[clause_k, lit_i])
            cb.set_ta_state(clause_k, lit_i, False, state[clause_k, lit_i + n_literals])


    cb.get_clause_state(dst, 1)

    assert (dst[0, :] == -69).all()
    assert (dst[-1, :] == -69).all()
    
    assert np.array_equal(dst[1:-1], state)
    cb.cleanup()


def test_CoaleasedTsetlinStateNV_set_clause_state():
    
    n_literals = 4
    n_classes = 2
    n_clauses = 3

    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    cb.initialize()
    dst = np.empty(shape=(n_clauses, n_literals*2), dtype=np.int8)

    src = np.array([[-1, -2, -3, -4, 1, 2, 3, 4],
                     [-10, -20, -30, -40, 10, 20, 30, 40],
                     [-11, -22, -33, -44, 11, 22, 33, 44]], dtype=np.int8)
    
    f = np.ones((n_clauses+2, n_literals*2), dtype=np.int8)
    f.fill(-103)
    f[1:-1, :] = src    
    f = np.ascontiguousarray(f)
    
    cb.set_clause_state(f, 1)
    cb.get_clause_state(dst, 0)

    assert np.array_equal(dst, src)
    cb.cleanup()
    

def test_CoaleasedTsetlinStateNV_get_clause_weights():
    n_literals = 2
    n_classes = 10
    n_clauses = 5
    n_extra = 2
    n_total_clauses = n_clauses+n_extra

    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    cb.initialize()
    
    src = np.arange(0, n_classes*n_total_clauses).astype(np.int16).reshape(n_total_clauses, n_classes)
    dst = np.empty_like(src)
    dst.fill(101)
        
    for clause_k in range(n_clauses):
        for class_i in range(n_classes):            
            cb.set_clause_weight(clause_k, class_i, src[clause_k, class_i])
            
    cb.get_clause_weights(dst, 1)

    assert (dst[0, :] == 101).all() 
    assert (dst[-1, :] == 101).all()        
    assert np.array_equal(dst[1:-1, :], src[0:n_clauses, :])
    cb.cleanup()
    
def test_CoaleasedTsetlinStateNV_set_clause_weights():
    n_literals = 2
    n_classes = 10
    n_clauses = 5
    n_extra = 2
    n_total_clauses = n_clauses+n_extra

    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    cb.initialize()
    
    src = np.arange(0, n_classes*n_total_clauses).astype(np.int16).reshape(n_total_clauses, n_classes)
    dst = np.empty_like(src)
    dst.fill(102)

    cb.set_clause_weights(src, 1)
    cb.get_clause_weights(dst, 1)
    
    assert (dst[0, :] == 102).all() 
    assert (dst[-1, :] == 102).all()    
    assert np.array_equal(dst[1:-1, :], src[1:n_clauses+1, :])
    cb.cleanup()
    

def test_SetClauseOutputNV_literal_counts_AND_force_one_true_literal_is_ON():
    n_literals = 4
    n_classes = 1
    n_clauses = 4
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNVPU(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)

    for c in range(n_clauses):
        for k in range(n_literals):        
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
            
    ib.prepare_example(0)
    gtc.time_train_set_clause_output_and_set_votes(cb)

    counts = cb.get_copy_literal_counts()
    assert counts[0] == 2 # positive
    assert counts[1] > 1000 # negated only, so since we are using PB - this will be a high number.
    assert counts[2] == 3 # pos+neg
    assert counts[3] == 0 # empty
    cb.cleanup()


def test_SetClauseOutputNV_literal_counts():
    n_literals = 4
    n_classes = 1
    n_clauses = 4
    threshold = 100
    s_param = 5.0
        
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)
    
    x = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    y = np.array([0], dtype=np.uint32)
    ib.set_data(x, y)

    for c in range(n_clauses):
        for k in range(n_literals):        
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
            
    ib.prepare_example(0)
    gtc.time_train_set_clause_output_and_set_votes(cb)

    counts = cb.get_copy_literal_counts()
    assert counts[0] == 2 # positive
    assert counts[1] == 1 # negated
    assert counts[2] == 3 # pos+neg
    assert counts[3] == 0 # empty
    cb.cleanup()


def test_xor_train_to_100_nv():
    n_literals = 10
    n_clauses = 16
    n_classes = 2
    s = 3.0
    n_literal_budget = 2
    threshold = 15    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=n_literal_budget)
    tm._tm_cls = gtc.ClauseBlockNV

    x, y, ex, ey = gt.dataset_generator.xor_dataset(noise=0.25, n_literals=n_literals, n_train=5000, n_test=500, seed=41)
    tm.set_train_data(x, y)
    tm.set_test_data(ex, ey)

    best_test_acc = 0.0
    for i in range(0, 5):
        trainer = gt.Trainer(threshold, n_epochs=100, seed=33+i, n_jobs=0, early_exit_acc=True, progress_bar=False)
        r = trainer.train(tm)
        best_test_acc = max(best_test_acc, r["best_test_score"])

        if r["did_early_exit"]:            
            break

    assert best_test_acc > 0.95




def weight_updates_correctly():
    n_literals = 1
    n_classes = 2
    n_clauses = 2
    threshold = 10
    s_param = 1.5
    n_literals_budget = 1
    
    ib = gtc.DenseInputBlock(n_literals)        
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes)
    feedback_block = gtc.FeedbackBlock(n_classes, threshold)
    
    cb.initialize()
    cb.set_feedback(feedback_block)
    cb.set_input_block(ib)
    cb.set_s(s_param)    
    cb.set_literal_budget(n_literals_budget)

    cb.set_ta_state(0, 0, True, 50)    
    cb.set_ta_state(0, 0, False, -51)

    cb.set_ta_state(1, 0, True, -52)    
    cb.set_ta_state(1, 0, False, 53)

    w = np.array([[-1, 1], [1, -1]], dtype=np.int16)
    cb.set_clause_weights(w, 0)

    x = np.array([[1], [0]], dtype=np.uint8)
    y = np.array([1, 0], dtype=np.uint32)
    ib.set_data(x, y)

    exec = gtc.SingleThreadExecutor([ib], [cb], feedback_block, 42)
    for _ in range(2):
        train_acc = exec.train_epoch()
        assert train_acc == 1.0

    c = np.empty(n_literals*2 * n_clauses, dtype=np.int8)
    cb.get_clause_state(c, 0)
    cb.get_clause_weights(w, 0)
    assert w[0,0] < -1
    assert w[0,1] > +1
    assert w[1,0] > +1
    assert w[1,1] < -1

    #print(c)
    #print(w)
    
    cb.cleanup()







if __name__ == "__main__":
    # test_ClauseBlockNV_init_clauses()
    # test_ClauseBlockNV_train_set_clause_output_and_set_votes()
    # test_Type2_feedback_ignore_positive_states()    
    # test_Type2_feedback_inc_negative_states_with_0_literal()
    # test_Type2_feedback_inc_negative_states_with_0_literal_stop_at_zero()

    # test_SetClauseOutputNV_literal_counts()
    # test_CoaleasedTsetlinStateNV_get_clause_state()
    # test_CoaleasedTsetlinStateNV_set_clause_state()
    
    # test_CoaleasedTsetlinStateNV_get_clause_weights()
    # test_CoaleasedTsetlinStateNV_set_clause_weights()

    # test_xor_train_to_100_nv()

    #test_Type1a_feedback_increase_states()
    # weight_updates_correctly()
    test_SetClauseOutputNV_literal_counts_AND_force_one_true_literal_is_ON()
    
    print("<tests - clause_block_nv - done>")







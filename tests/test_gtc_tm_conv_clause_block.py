
import numpy as np
import pytest 
import green_tsetlin_core as gtc 
import green_tsetlin as gt


def test_set_init_fails_with_no_patch_size_set():
    n_literals = 12
    n_clauses = 42 
    n_classes = 3       

    cb_no_patch = gtc.ClauseBlockConvTM(n_literals, n_clauses, n_classes)    
    assert cb_no_patch.initialize(seed=42) == False

    cb_zero_patch = gtc.ClauseBlockConvTM(n_literals, n_clauses, n_classes)
    cb_zero_patch.set_number_of_patches_per_example(0)
    assert cb_zero_patch.initialize(seed=42) == False

    cb_patch_set = gtc.ClauseBlockConvTM(n_literals, n_clauses, n_classes)
    cb_patch_set.set_number_of_patches_per_example(1)
    assert cb_patch_set.initialize(seed=42)


def test_train_simple_xor():
    

    def inject_conv_tm(a,b,c):
        cb = gtc.ClauseBlockConvTM(a,b,c)
        cb.set_number_of_patches_per_example(1)
        return cb

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    

    tm_flat = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)         
    tm_flat._backend_clause_block_cls = gtc.ClauseBlockTM_Lt_Bt
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm_flat, seed=32, n_jobs=1, n_epochs=100)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()    
    assert r["did_early_exit"]
    
    state0 = tm_flat._state.copy()

    tm_conv = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm_conv._backend_clause_block_cls = inject_conv_tm
    #tm_conv._backend_clause_block_cls = gtc.ClauseBlockTM
    #tm_conv.load_state(state0)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm_conv, seed=32, n_jobs=1, n_epochs=100)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()    
    #assert r["did_early_exit"]
    print(r)


    # tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    # tm._backend_clause_block_cls = inject_conv_tm
    
    # x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    # trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=100)
    # trainer.set_train_data(x, y)
    # trainer.set_test_data(ex, ey)
    # r = trainer.train()    
    # print(r)
    
    #print(tm._state.c)
    #print(tm._state.w)
    #

    



if __name__ == "__main__":
    test_set_init_fails_with_no_patch_size_set()
    test_train_simple_xor()

    print("<done:", __file__, ">")

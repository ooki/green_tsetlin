import numpy as np
import pytest 
import green_tsetlin_core as gtc 
import green_tsetlin as gt


if gtc.has_avx2() is False:
    pytest.skip(allow_module_level=True)



def test_train_simple_xor():
    

    def inject_conv_tm(a,b,c):
        #cb = gtc.ClauseBlockConvTM(a,b,c)
        cb = gtc.ClauseBlockConvAVX2(a,b,c)
        cb.set_number_of_patches_per_example(1)
        return cb

    n_literals = 7
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42


    tm_conv = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm_conv._backend_clause_block_cls = inject_conv_tm
    #tm_conv._backend_clause_block_cls = gtc.ClauseBlockTM
    #tm_conv.load_state(state0)
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals)    
    trainer = gt.Trainer(tm_conv, seed=35, n_jobs=1, n_epochs=100)
    trainer.set_train_data(x, y)
    trainer.set_eval_data(ex, ey)
    r = trainer.train()
    print(r)
    

if __name__ == "__main__":
    test_train_simple_xor()
    print("<done:", __file__, ">")

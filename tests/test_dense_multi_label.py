
import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc 




def test_multi_label_xor_gtc():
    n_literals = 5
    
    n_train = 1000
    n_test = 100
    
    n_classes = 2
    noise = 0.3
    n_clauses = 6
    s_param = 3.0
    threshold = 25
    n_literal_budget = 2
    seed = 42
    n_epochs = 100
    
    xt, yt, xe, ye = gt.dataset_generator.multi_label_xor(noise=noise, n_train=n_train, n_test=n_test, n_literals=n_literals, n_classes=n_classes, seed=42)
    ib = gtc.DenseInputBlock(n_literals)
    ib.set_data(xt, yt)    
    mlfb = gtc.FeedbackBlockMultiLabel(n_classes, threshold, seed)
    
    cb = gtc.ClauseBlockNV(n_literals, n_clauses, n_classes*2) # x2 since we are binary multi labeling    
    cb.set_s(s_param)
    cb.set_literal_budget(n_literal_budget)
    
    cb.set_input_block(ib)
    cb.set_feedback(mlfb)
    cb.initialize(42)
    
    
    exec = gtc.SingleThreadExecutor([ib], [cb], mlfb, seed)
    
    is_solved = False
    for epoch in range(n_epochs):        
        ib.set_data(xt, yt) 
        
        train_acc = exec.train_epoch()
        
        ib.set_data(xe, ye)
        y_hat = np.array(exec.eval_predict_multi())
        
        if  (y_hat == ye).mean() > 0.999:
            is_solved = True        
            break
        
    cb.cleanup()
    
    assert is_solved
    


if __name__ == "__main__":
    test_multi_label_xor_gtc()
    print("<done tests:", __file__, ">")
    



import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc 



def test_multi_label_xor_trainer():
    n_train = 1000
    n_test = 100
    
    n_classes = 4
    n_literals = n_classes+1
    noise = 0.2
    n_clauses = 30
    s_param = 3.0
    threshold = 50
    n_literal_budget = 3
    seed = 42
    n_epochs = 100
    n_jobs = 2

    xt, yt, xe, ye = gt.dataset_generator.multi_label_xor(noise=noise, n_train=n_train, n_test=n_test, n_literals=n_literals, n_classes=n_classes, seed=42)

    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_param, n_literal_budget=n_literal_budget, multi_label=True)
    tm.set_train_data(xt, yt)
    tm.set_test_data(xe, ye)

    trainer = gt.Trainer(threshold=threshold, progress_bar=False, n_epochs=n_epochs,  n_jobs=n_jobs, seed=seed)
    r = trainer.train(tm)
    d = tm.get_state()

    assert d["w"].shape[0] == n_clauses and d["w"].shape[1] == (n_classes*2) # 0/1 per class
    print("best:",  r["best_test_score"])
    assert r["best_test_score"] > 0.9



def test_multi_label_xor_gtc():
    
    
    n_train = 1000
    n_test = 100
    
    n_classes = 3
    n_literals = n_classes+1
    noise = 0.1
    n_clauses = 6
    s_param = 3.0
    threshold = 25
    n_literal_budget = 2
    seed = 42
    n_epochs = 100
    n_jobs = 2

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
    # exec = gtc.MultiThreadExecutor([ib], [cb], mlfb, n_jobs, seed)
    
    is_solved = False
    for epoch in range(n_epochs):        
        ib.set_data(xt, yt) 
        
        train_acc = exec.train_epoch()
        
        ib.set_data(xe, ye)
        y_hat = np.array(exec.eval_predict_multi())
        
        if  (y_hat == ye).mean() > 0.9:
            is_solved = True        
            break
        
    cb.cleanup()
    
    assert is_solved
    


if __name__ == "__main__":
    #test_multi_label_xor_gtc()
    test_multi_label_xor_trainer()
    print("<done tests:", __file__, ">")
    


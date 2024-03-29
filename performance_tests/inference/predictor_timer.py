from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
import tqdm

import green_tsetlin as gt



if __name__ == "__main__":
    
    x = np.fromfile("mnist_x_70000_784.test_bin", dtype=np.uint8).reshape(-1, 784)
    y = np.fromfile("mnist_y_70000_784.test_bin", dtype=np.uint32).reshape(x.shape[0])
        
    ds = gt.DenseState.load_from_file("mnist_state.npz")    
    rs = gt.RuleSet(is_multi_label=False)
    rs.compile_from_dense_state(ds)
    
    predictor = gt.Predictor.from_ruleset(rs)
    
    # warmup 
    warmup_x = np.zeros(shape=x.shape[1], dtype=np.uint8)
    y_warmup = predictor.predict(warmup_x)
    
    total_time = 0.0
    n_total_examples = 10_000
    n_total_correct = 0
    print("starting predictions.")
    for k in tqdm.tqdm(range(n_total_examples)):
        i = k % x.shape[0]
        ex = x[i]
        ex_y = y[i]
        
        t0 = perf_counter()
        y_hat = predictor.predict(ex)
        t1 = perf_counter()
        
        if y_hat == ex_y:
            n_total_correct += 1
        
        total_time += (t1 - t0)
        
    print("Results:")
    print("Total time:", total_time)
    print("n_total_examples:", n_total_examples)
    print("n_total_correct:", n_total_correct)
    
    print("<done>")
        
        
    
    
    
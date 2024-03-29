from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

import green_tsetlin as gt
import green_tsetlin_core as gtc

if __name__ == "__main__":
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    x, y = shuffle(X, y, random_state=42)  
    x = np.where(x.reshape((x.shape[0], 28 * 28)) > 75, 1, 0)
    x = x.astype(np.uint8)
    y = y.astype(np.uint32)
    
    n_examples = x.shape[0]
    n_literals = x.shape[1]      
    x.astype(np.uint8).tofile("./generator_tests/mnist_x_{}_{}.test_bin".format(n_examples, n_literals))
    y.astype(np.uint32).tofile("./generator_tests/mnist_y_{}_{}.test_bin".format(n_examples, n_literals))
    
    n_clauses = 1000
    n_literals = x.shape[1]
    n_classes = 10
    s = 10.0
    n_literal_budget = 8
    threshold = 1000    
    n_jobs = 2
    seed = 42
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s,
                           threshold=threshold, literal_budget=n_literal_budget)
    
    # ------
    trainer = gt.Trainer(tm, n_epochs=3, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(x, y)
    trainer.set_test_data(x.copy(), y.copy())
    trainer.train()        
    tm.save_state("./generator_tests/mnist_state.npz")
    
    print("--- results ---")
    print(trainer.results)
    print("--")
    
    rs = tm.get_ruleset()
    writer = gt.Writer(rs)
    writer.to_file("./generator_tests/mnist_tm.h")
    
    did_print = False
    
    correct = 0
    correct2 = 0
    total = 0
    p = tm.get_predictor()
    for k in range(0, x.shape[0]):
        y_hat = p.predict(x[k, :])
        if y_hat == y[k]:
            correct += 1
            
        total += 1
        
    print("correct:", correct, "correct2:", correct2, "total:", total)        
    print("<done>")
    
    
    
    
    
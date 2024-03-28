from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

import green_tsetlin as gt

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
    x.astype(np.uint8).tofile("./generator_tests/mnist_x_{}_{}.bin".format(n_examples, n_literals))
    y.astype(np.uint32).tofile("./generator_tests/mnist_y_{}_{}.bin".format(n_examples, n_literals))
    
    n_clauses = 100
    n_literals = x.shape[1]
    n_classes = 10
    s = 10.0
    n_literal_budget = 5
    threshold = 1000    
    n_jobs = 2
    seed = 42
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget)
    
    trainer = gt.Trainer(tm, n_epochs=2, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(x, y)
    trainer.set_test_data(x.copy(), y.copy())
    trainer.train()        
    tm.save_state("./generator_tests/mnist_state.npz")
    
    rs = tm.get_ruleset()
    writer = gt.Writer(rs)
    writer.to_file("./generator_tests/mnist_tm.h")
    
    print("<done>")
    
    
    
    
    
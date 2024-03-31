from time import perf_counter
import random
import pickle
import os 

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image as im
import tqdm

import green_tsetlin as gt

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


def get_mnist():
    t0 = perf_counter()
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=10_000)
    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = X_train.astype(np.uint8)
                
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = X_test.astype(np.uint8)
    
    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    
    t1 = perf_counter()    
    delta = t1 - t0
    print("getting data time:{}".format(delta))

    return X_train, X_test, y_train, y_test


def log_result(s):
    with open("mnist_performance_single.log", "a") as fp:
        fp.write("{}\n".format(s))

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = get_mnist()
    
    s = 10.0
    n_clauses = 5_000
    threshold = n_clauses // 4
    
    all_backends = ["pyTsetlinMachine", "gt_job1", "tmu"]
    
    #backend_to_test = 
    # backend_to_test = "gt_job1"
    
    log_result("-- new run, clauses:{} --".format(n_clauses))
    
    # for backend_to_test in ["pyTsetlinMachine"]:
    for backend_to_test in all_backends:
    
        print("TESTING BACKEND:", backend_to_test)
        
        t0_total = perf_counter()
        
        if backend_to_test == "tmu":
            tm = TMCoalescedClassifier(
                number_of_clauses=n_clauses,
                T=threshold,
                s=s,
                weighted_clauses=True,
                focused_negative_sampling=True,
                platform="CPU"
            )
            
            
            for epoch in range(5):
                #t0 = perf_counter()
                tm.fit(X_train, y_train)
                result = 100 * (tm.predict(X_test) == y_test).mean()            
                print("result:", result)
                #t1 = perf_counter()
                #print("epoch [{}] time: {:.3f}".format(epoch, t1 - t0) )
                
        
        elif backend_to_test == "gt_job1":
            tm = gt.TsetlinMachine(n_literals=X_train.shape[1], n_clauses=n_clauses, n_classes=10, s=s, threshold=threshold, literal_budget=30)
            
            trainer = gt.Trainer(tm, n_epochs=5, seed=42, n_jobs=1, progress_bar=True, early_exit_acc=2.0)
            trainer.set_train_data(X_train, y_train)
            trainer.set_test_data(X_test, y_test)
            
            r = trainer.train()
            print(r)
            
            
        elif backend_to_test == "pyTsetlinMachine":
            tm = MultiClassTsetlinMachine(number_of_clauses=n_clauses, T=threshold, s=s, boost_true_positive_feedback=1, weighted_clauses=True, max_included_literals=30)

            for i in range(5):
                tm.fit(X_train, y_train, epochs=1, incremental=True)
                result = 100*(tm.predict(X_test) == y_test).mean()
                print("#{} Accuracy: {:.2f} ".format(i+1, result))
        
        t1_total = perf_counter()
        delta_t = "total time: {:.3f}".format(t1_total - t0_total)
        
        log_result("{} => {} secs".format(backend_to_test, delta_t))
        print("done: {} => {} secs".format(backend_to_test, delta_t))
        
    print("<done>")
    
    
    """
    tmu:
    epoch [0] time: 125.900
    epoch [1] time: 95.401
    epoch [2] time: 89.209
    epoch [3] time: 83.345
    epoch [4] time: 82.660
    
    gt1 : 114.00 (all 5)

    MultiClassTsetlinMachine
    total time: 206.603
    """
    








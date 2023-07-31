from time import perf_counter


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image as im


import green_tsetlin as gt 



def get_mnist():
    t0 = perf_counter()
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=10000)
    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_train = X_train.astype(np.uint8)
        
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = X_test.astype(np.uint8)
    
    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    
    t1 = perf_counter()    
    delta = t1 - t0
    print("getting data time:{}".format(delta))

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    xt, xe, yt, ye = get_mnist()
    
    print("xt:", xt.shape)
    print("xe:", xe.shape)

    n_literals = 784
    n_clauses = 20000 // 5
    n_classes = 10
    s = 10.0
    n_literal_budget = 20
    threshold = 1000    
        
    log = {}
    n_repeats = 5
    
    for n_jobs in range(0, 15):
        results = []
        for _ in range(n_repeats):        
            tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=n_literal_budget)
            tm.set_train_data(xt, yt)
            tm.set_test_data(xe, ye)
            trainer = gt.Trainer(threshold, n_epochs=1, seed=32, n_jobs=n_jobs, early_exit_acc=True)
            
            t0 = perf_counter()
            r = trainer.train(tm)
            t1 = perf_counter()
            
            results.append(t1 - t0)
            
        log[n_jobs] = np.median(results)
        
    print("--- done ---")
    print(log)

        
    """_c:1000
    {0: 6.868921039975248, 1: 6.886135007021949, 2: 8.663222913979553, 3: 7.246398693008814, 4: 6.549179043970071, 5: 6.119028838002123, 6: 5.909698552975897, 7: 5.85849223198602, 8: 5.731698243005667, 9: 5.643217025033664, 10: 5.642465525015723, 11: 5.754526455013547, 12: 6.491689677000977, 13: 6.905591208022088}
    
    c: 4000
    {0: 25.424992660002317, 1: 25.496304271975532, 2: 22.096862260019407, 3: 19.424868742993567, 4: 16.46304657298606, 5: 14.570514718012419, 6: 13.605244976992253, 7: 12.870885292009916, 8: 12.249741650011856, 9: 11.957188666972797, 10: 11.718457520008087, 11: 11.681772134965286, 12: 13.70202619000338, 13: 13.714507074037101}
    
    """
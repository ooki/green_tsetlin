from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import tqdm

import green_tsetlin as gt

def get_mnist():
    t0 = perf_counter()
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=1)
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
    
    n_clauses = 5000
    n_literals = xt.shape[1]
    n_classes = 10
    s = 10.0
    n_literal_budget = 10
    threshold = 1000    
    n_jobs = 1
    seed = 42
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget)
    
    trainer = gt.Trainer(tm, n_epochs=1, seed=seed, n_jobs=n_jobs, progress_bar=False)
    trainer.set_train_data(xt, yt)
    trainer.set_test_data(xe, ye)    
    trainer.train()
    
    print("<done>")
    
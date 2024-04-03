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

from scipy.sparse import csr_matrix

def get_mnist():
    t0 = perf_counter()
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
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


def run_dense(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs):
    xt, xe, yt, ye = get_mnist()
    
    n_literals = xt.shape[1]
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget)
    
    trainer = gt.Trainer(tm, n_epochs=n_epochs, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(xt, yt)
    trainer.set_test_data(xe, ye)    
    trainer.train()
    
    out_file = "mnist_state.npz"
    tm.save_state(out_file)
    
    print("saved state to: '{}'".format(out_file))
    print("Result:")
    for k, v in trainer.results.items():
        print("[{}] = {}".format(k,v))

def run_sparse(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs):
    xt, xe, yt, ye = get_mnist()

    xt = csr_matrix(xt)
    xe = csr_matrix(xe)
    
    n_literals = xt.shape[1]
    
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget, boost_true_positives=True, dynamic_AL=True)
    tm.active_literals_size = 60
    tm.clause_size = 28
    tm.lower_ta_threshold = -40


    trainer = gt.Trainer(tm, n_epochs=n_epochs, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(xt, yt)
    trainer.set_test_data(xe, ye)    
    trainer.train()
    
    out_file = "mnist_state_sparse.npz"
    tm.save_state(out_file)
    
    print("saved state to: '{}'".format(out_file))
    print("Result:")
    for k, v in trainer.results.items():
        print("[{}] = {}".format(k,v))

if __name__ == "__main__":
    
    n_clauses = 10000
    n_classes = 10
    s = 10.0
    n_literal_budget = 20
    threshold = 1000
    n_jobs = 3
    seed = 42
    n_epochs = 3

    # run_dense(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs)
    run_sparse(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs)
    
    print("<done>")
    
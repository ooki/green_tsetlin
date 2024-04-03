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

def get_imdb():

    

    return X_train, X_test, y_train, y_test


def run_dense(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs):
    xt, xe, yt, ye = get_imdb()
    
    n_literals = xt.shape[1]
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget)
    
    trainer = gt.Trainer(tm, n_epochs=n_epochs, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(xt, yt)
    trainer.set_test_data(xe, ye)    
    trainer.train()
    
    out_file = "imdb_state.npz"
    tm.save_state(out_file)
    
    print("saved state to: '{}'".format(out_file))
    print("Result:")
    for k, v in trainer.results.items():
        print("[{}] = {}".format(k,v))

def run_sparse(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs):
    xt, xe, yt, ye = get_imdb()

    xt = csr_matrix(xt)
    xe = csr_matrix(xe)
    
    n_literals = xt.shape[1]
    
    tm = gt.SparseTsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget, boost_true_positives=True, dynamic_AL=True)
    tm.active_literals_size = 120
    tm.clause_size = 80
    tm.lower_ta_threshold = -40

    trainer = gt.Trainer(tm, n_epochs=n_epochs, seed=seed, n_jobs=n_jobs, progress_bar=True)
    trainer.set_train_data(xt, yt)
    trainer.set_test_data(xe, ye)    
    trainer.train()
    
    out_file = "imdb_state_sparse.npz"
    tm.save_state(out_file)
    
    print("saved state to: '{}'".format(out_file))
    print("Result:")
    for k, v in trainer.results.items():
        print("[{}] = {}".format(k,v))


if __name__ == "__main__":
    
    n_clauses = 1000
    n_classes = 2
    s = 2.0
    n_literal_budget = 20
    threshold = 2000
    n_jobs = 3
    seed = 42
    n_epochs = 3

    run_dense(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs)
    run_sparse(n_clauses, n_classes, s, n_literal_budget, threshold, n_jobs, seed, n_epochs)
    
    print("<done>")
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

try:
    import cpuinfo
    from cpuinfo import get_cpu_info
    get_cpu_info()
    
except (ImportError, ModuleNotFoundError):
    print("Please install 'pip install py-cpuinfo' to run this test.")
    exit(0)

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


def get_data_selection():
    xt1, xe1, yt, ye = get_mnist()

    xt2 = np.concatenate([xt1, xt1], axis=1)
    xe2 = np.concatenate([xe1, xe1], axis=1)    
    
    n = xt1.shape[1] // 2
    d = xt1.shape[0] // 2
    
    xt05 = xt1[:, 0:n]
    xe05 = xe1[:, 0:n]
    
    
    # make bigger    
    xt05_small = xt05[0:d, :]        
    xt1_small = xt1[0:d, :]
    xt2_small = xt2[0:d, :]
    ye_small = ye[0:d]
    yt_small = yt[0:d]
    
        
    xt05_big = np.concatenate([xt05, xt05_small], axis=0)
    xt1_big = np.concatenate([xt1, xt1_small], axis=0)
    xt2_big = np.concatenate([xt2, xt2_small], axis=0)
    
    yt_big = np.concatenate([yt, yt_small], axis=0)    
    ye_big = np.concatenate([ye, ye_small], axis=0)
    
    
    
    data = [(xt05_small, xe05, yt_small, ye_small),
            (xt05,       xe05, yt, ye),
            (xt05_big,   xe05, yt_big, ye_big),
            (xt1_small,  xe1,  yt_small, ye_small),
            (xt1_small,  xe1,  yt, ye),
            (xt1_big,    xe1,  yt_big, ye_big),
            (xt2_small,  xe2,  yt_small, ye_small),
            (xt2,        xe2,  yt, ye),
            (xt2_big,    xe2,  yt_big, ye_big)]
    
    return data


def run_trial(data, seed):
    
    # random params    
    data_size = random.randint(0, len(data)-1)
    # testing
    train_x, test_x, train_y, test_y = data[data_size]    
    n_literals = train_x.shape[1]        
    n_clauses = random.randint(50, 20000)
        
    n_jobs = random.randint(1, 24)
    
    min_cbs = max(n_jobs-1, 1)
    n_cbs = random.randint(min_cbs, n_jobs*3)
    
    # static    
    n_classes = 10
    s = 10.0
    n_literal_budget = 20
    threshold = 1000    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=n_literal_budget)
    tm.set_num_clause_blocks(n_cbs)
    
    trainer = gt.Trainer(tm, n_epochs=1, seed=seed, n_jobs=n_jobs, progress_bar=False)
    trainer.set_train_data(train_x, train_y)
    trainer.set_test_data(test_x, test_y)
    
    t0 = perf_counter()
    r = trainer.train()
    t1 = perf_counter()
    
    training_time = t1 - t0
    
    return {
        "data_size": data_size,
        "n_literals": n_literals,
        "n_examples": train_x.shape[0],
        "n_clauses": n_clauses,
        "n_cbs": n_cbs,
        "n_jobs": n_jobs,
        "time": training_time,
        "train_accuracy": r["train_log"][0]
    }
    

def inf_gen():
    i = 0
    while True:
        yield i
        i += 1

if __name__ == "__main__":
    
    data = get_data_selection()
    log = None
    seed = 42
    
    out_file = "mnist_dense_results.pkl"
    
    for _ in tqdm.tqdm(inf_gen()):        
        if os.path.isfile(out_file):
            if log is None:
                print("Log exist : adding to it")
                
            with open(out_file, "rb") as fp:
                log = pickle.load(fp)
                
        else:
            print("No log file found, creating new log.")
            ci = get_cpu_info()
            log = [ci]
                
            
        r = run_trial(data, seed)
        seed += 1
        log.append(r)
        
        with open(out_file, "wb") as fp:
            pickle.dump(log, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
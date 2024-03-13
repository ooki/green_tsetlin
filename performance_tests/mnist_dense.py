from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from PIL import Image as im
import tqdm

try:
    import cpuinfo
    from cpuinfo import get_cpu_info    
    
except (ImportError, ModuleNotFoundError):
    print("Please install 'pip install py-cpuinfo' to run this test.")
    exit(1)


try:
    import psutil
except (ImportError, ModuleNotFoundError):
    print("Please install 'pip install psutil' to run this test.")
    exit(2)

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


def get_big_data_block():
    xt1, xe1, yt, ye = get_mnist()

    xt3 = np.concatenate([xt1, xt1, xt1], axis=1)
    big_x = np.concatenate([xt3, xt3], axis=0)
    big_y = np.concatenate([yt, yt], axis=0)

    return big_x, big_y



def run_trial(data, seed):
    
    data_x, data_y = data

    n_literals = random.randint(2, data_x.shape[1]-1)
    n_examples = random.randint(2, data_x.shape[0]-1)
    n_clauses = random.randint(10, 20000)
    
    n_jobs = random.randint(1, 24)
    min_cbs = max(n_jobs-1, 1)
    n_cbs = random.randint(min_cbs, n_jobs*3)

    train_x = data_x[:n_examples, :n_literals]
    train_y = data_y[:n_examples]

    test_x = data_x[0:2, :n_literals]
    test_y = data_y[0:2]
    
    # static    
    n_classes = 10
    s = 10.0
    n_literal_budget = 30
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
        "n_literals": n_literals,
        "n_examples": n_examples,
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
    
    n_fails_left = 3
    cpu_info = get_cpu_info()
    computer_id = uuid.UUID(int=uuid.getnode()).int
    out_file = "./results/mnist_dense_results_{}.pkl".format(computer_id)

    print("-"*80)
    print()
    print("PLEASE MAKE SURE YOU DONT RUN ANY OTHER HEAVY PROCESSES WHILE RUNNING THE TESTS!!")
    print("Also make sure you run in from the 'performance_tests' folder!")
    print()
    print("id: {}".format(computer_id))
    print("out: {}".format(out_file))
    print()
    print("-"*80)

    data = get_big_data_block()    
    
    log = None
    seed = 42
    
    for _ in tqdm.tqdm(inf_gen()):        
        if os.path.isfile(out_file):
            if log is None:
                print("Log exist : adding to it")
                
            with open(out_file, "rb") as fp:
                log = pickle.load(fp)
                
        else:
            print("No log file found, creating new log.")
            log = {}
            log["cpu"] = cpu_info
            log["physical_cores"] = psutil.cpu_count(logical=False)
            log["logical_cores"] = psutil.cpu_count(logical=True)
            log["ram"] = psutil.virtual_memory().total / 1e9
            log["trials"] = []

            # early fail if we cant write to the file.
            with open(out_file, "wb") as fp:
                pickle.dump(log, fp, protocol=pickle.HIGHEST_PROTOCOL)

                
        try:
            seed += 1
            r = run_trial(data, seed)
            log["trials"].append(r)
        except:
            n_fails_left -= 1
            if n_fails_left == 0:
                print("Too many failures, exiting.")
                break
        
        with open(out_file, "wb") as fp:
            pickle.dump(log, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
import tqdm

import green_tsetlin as gt



if __name__ == "__main__":
    ds = gt.DenseState.load_from_file("mnist_state.npz")    
    rs = gt.RuleSet(is_multi_label=False)
    rs.compile_from_dense_state(ds)
    
    
    w = gt.Writer(rs)
    w.to_file("mnist_tm.h")
    
    print("<done>")
    
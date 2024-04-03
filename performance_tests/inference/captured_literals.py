from time import perf_counter
import random
import pickle
import os 
import uuid

import numpy as np
import tqdm

import green_tsetlin as gt
from collections import Counter
from itertools import chain



if __name__ == "__main__":
    dense_state = gt.DenseState.load_from_file("mnist_state.npz")
    sparse_state = gt.SparseState.load_from_file("mnist_state_sparse.npz")

    dense_rs = gt.RuleSet(is_multi_label=False)
    dense_rs.compile_from_dense_state(dense_state)
    sparse_rs = gt.RuleSet(is_multi_label=False)
    sparse_rs.compile_from_sparse_state(sparse_state)

    # print(dense_rs.rules)
    # print(sparse_rs.rules)

    dense_counter = Counter(chain(*dense_rs.rules))
    sparse_counter = Counter(chain(*sparse_rs.rules))


    print("Dense literals: ", len(dense_counter))  # has all 1568 literals in the rules
    # print(dense_counter)
    print("Sparse literals: ", len(sparse_counter)) # has only 861 literals in the rules
    # print(sparse_counter)



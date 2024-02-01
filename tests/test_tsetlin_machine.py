from collections.abc import Iterable
import random
import math

import pytest
import numpy as np


import green_tsetlin as gt


def test_throws_on_invalid_s():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s_invalid = 0.23
    s_valid = 1.0
    threshold = 10
    
    gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold)
    with pytest.raises(ValueError):
        gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_invalid, threshold=threshold)

def test_throws_on_invalid_threshold():
    
    n_literals = 7
    n_clauses = 12
    n_classes = 3
    s_valid = 1.0
    threshold_valid = 10
    threshold_invalid = -2
    
    gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold_valid)
    with pytest.raises(ValueError):
        gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s_valid, threshold=threshold_invalid)


from collections import namedtuple

import pytest
import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc


def test_init():
    n_literals = 4
    n_clauses = 5
    n_classes = 2
    s = 3.0
    threshold = 42    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        
    tm._backend_clause_block_cls = gtc.ClauseBlockTM
    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, noise=0.05)    
    trainer = gt.Trainer(tm, seed=32, n_jobs=1)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    trainer.train()
    
    predictor = gt.Predictor.create(tm)
    

    
    
    
    



if __name__ == "__main__":
    test_init()
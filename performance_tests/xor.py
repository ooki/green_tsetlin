from time import perf_counter
import random
import pickle
import os 
import uuid


import green_tsetlin as gt 




if __name__ == "__main__":
    n_literals = 12
    n_clauses = 10
    n_classes = 2
    s = 3.5
    threshold = 50
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)        

    
    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, noise=0.25, n_train=1000, n_test=100)
    trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=20, early_exit_acc=1.5)
    trainer.set_train_data(x, y)
    trainer.set_test_data(ex, ey)
    trainer.train()    

    print(trainer.results)

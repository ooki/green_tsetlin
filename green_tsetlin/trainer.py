import os
from typing import Optional, List, Union
from collections import defaultdict

    
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm


#import green_tsetlin_core as gtc
from green_tsetlin import TsetlinMachine

def empty_epoch_callback(epoch, train_acc, test_score):
    pass


class Trainer:
    def __init__(self,
                 tm: TsetlinMachine,
                 feedback_type="focused_negative_sampling",
                 n_jobs:int=1,
                 n_epochs: int = 20,
                 seed:int=0,
                 early_exit_acc:float=1.0,
                 load_best_state:bool=True,
                 fn_epoch_callback=empty_epoch_callback,
                 fn_test_score="accuracy",
                 progress_bar=True,
                 copy_training_data:bool=True):

        self.tm = tm
        self.feedback_type = feedback_type
        self.seed = seed
        self.n_epochs = n_epochs
        self.load_best_state = load_best_state
        self.early_exit_acc = early_exit_acc
        self.progress_bar = progress_bar
        
        self.fn_epoch_callback = fn_epoch_callback        
        self.fn_test_score = fn_test_score

        self.copy_training_data = copy_training_data
        
        if n_jobs < 0:
            n_cpus = os.cpu_count()                        
            self.n_jobs = n_cpus + 1 + n_jobs # remember: n_jobs is negative
            
        elif n_jobs == 0:
            self.n_jobs = 1
            
        else:
            self.n_jobs = n_jobs


    def set_data(self, x_train:np.array, y_train:np.array, x_test:np.array=None, y_test:np.array=None):

        if x_test is not None and y_test is None:
            raise ValueError("y_test must be provided if x_test is provided")

        if self.copy_training_data:
            x_train = x_train.copy()
            y_train = y_train.copy()

            if x_test is not None:
                x_test = x_test.copy()
                y_test = y_test.copy()
        
        if x_train.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(x_train.dtype))
        
        if x_test is not None and x_test.dtype != np.uint8:
            raise ValueError("Data x_test must be of type np.uint8, was: {}".format(x_test.dtype))

        if y_train.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(y_train.dtype))

        if y_test is not None and y_test.dtype != np.uint32:
            raise ValueError("Data y_test must be of type np.uint32, was: {}".format(y_test.dtype))

        if x_train.shape[1] != self.tm.n_literals:
            raise ValueError("Data x_train does not match in shape[1] (#literals) with n_literals : {} != {}".format(x_train.shape[1], self.tm.n_literals))

        if x_test is not None and x_test.shape[1] != self.tm.n_literals:
            raise ValueError("Data x_test does not match in shape[1] (#literals) with n_literals : {} != {}".format(x_test.shape[1], self.tm.n_literals))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test



    def train(self):
        pass

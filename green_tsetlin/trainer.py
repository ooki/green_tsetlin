import os
from typing import Optional, List, Union
from collections import defaultdict

    
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm



#import green_tsetlin_core as gtc
from green_tsetlin import TsetlinMachine, TMState, allocate_clause_blocks
from green_tsetlin import py_gtc

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

        self._best_tm_state : TMState = None


        if fn_test_score == "accuracy":
            self.fn_test_score = accuracy_score        
        else:
            self.fn_test_score = fn_test_score
        
        if n_jobs < 0:
            n_cpus = os.cpu_count()                        
            self.n_jobs = n_cpus + 1 + n_jobs # remember: n_jobs is negative
            
        elif n_jobs == 0:
            self.n_jobs = 1
            
        else:
            self.n_jobs = n_jobs

        # backend setup
        self._cls_dense_ib = py_gtc.DenseInputBlock        
        self._cls_feedback_block = py_gtc.FeedbackBlock
        self._cls_feedback_block_multi_label = py_gtc.FeedbackBlockMultiLabel
        self._cls_exec_singlethread = py_gtc.SingleThreadExecutor
        self._cls_exec_multithread = py_gtc.MultiThreadExecutor

    def set_train_data(self, x_train:np.array, y_train:np.array):
        
        y_train = np.atleast_1d(y_train)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Data x_train and y_train must have the same number of examples: {} != {}".format(x_train.shape[0], y_train.shape[0]))

        if self.copy_training_data:
            x_train = x_train.copy()
            y_train = y_train.copy()
           
        if x_train.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(x_train.dtype))
        
        if y_train.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(y_train.dtype))

        if x_train.shape[1] != self.tm.n_literals:
            raise ValueError("Data x_train does not match in shape[1] (#literals) with n_literals : {} != {}".format(x_train.shape[1], self.tm.n_literals))


        self.x_train = x_train
        self.y_train = y_train        

    def set_test_data(self, x_test:np.array, y_test:np.array):
        
        y_test = np.atleast_1d(y_test)
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError("Data x_test and y_test must have the same number of examples: {} != {}".format(x_test.shape[0], y_test.shape[0]))
             
        if x_test is not None:
            x_test = x_test.copy()
            y_test = y_test.copy()

        if x_test.dtype != np.uint8:
            raise ValueError("Data x_test must be of type np.uint8, was: {}".format(x_test.dtype))
        
        if y_test.dtype != np.uint32:
            raise ValueError("Data y_test must be of type np.uint32, was: {}".format(y_test.dtype))
        
        if x_test.shape[1] != self.tm.n_literals:
            raise ValueError("Data x_test does not match in shape[1] (#literals) with n_literals : {} != {}".format(x_test.shape[1], self.tm.n_literals))
        
        self.x_test = x_test
        self.y_test = y_test


    def _get_feedback_block(self, n_classes, threshold):
        if self.feedback_type == "focused_negative_sampling":
            if self.tm._is_multi_label is False:

                return self._cls_feedback_block(n_classes, threshold, self.seed)        
            else:
                # since we have actual classes as 2*classes since we use 0/1 per class
                return self._cls_feedback_block_multi_label(n_classes // 2, threshold, self.seed)
            
        
        else:
            raise ValueError("Unknown feedback type: {}".format(self.feedback_type))
        
        
    def _calculate_blocks_for_tm(self):
        return self.n_jobs
    
    def train(self):                
        
        ib = self._cls_dense_ib(self.tm.n_literals)
        ib.set_data(self.x_train, self.y_train)
        feedback_block = self._get_feedback_block(self.n_classes, self.threshold)
        
        n_blocks = self._calculate_blocks_for_tm()
        cbs = self.tm.construct_clause_blocks(n_blocks)
        
        
        if self.n_epochs < 1:            
            return
        
        if self.n_jobs == 1:
            exec = self._cls_exec_singlethread(ib, cbs, feedback_block, self.seed)
        else:
            exec = self._cls_exec_multithread(ib, cbs, feedback_block, self.n_jobs, self.seed)
        
        
        
        # main loop
        n_epochs_trained = 0
        best_test_score = -1.0
        best_test_epoch = -1
        train_acc = -1.0
        
        
        train_log = []
        test_log = []
        did_early_exit = False


        with allocate_clause_blocks(cbs, seed=self.seed):        
            hide_progress_bar = self.progress_bar is False  
            with tqdm.tqdm(total=self.n_epochs, disable=hide_progress_bar) as progress_bar:
                progress_bar.set_description("Processing epoch 1 of {}, train acc: NA, best test score: NA".format(self.n_epochs))
        
                for epoch in range(self.n_epochs):                                            
                    train_acc = exec.train_epoch()
                    train_log.append(train_acc)                
                    n_epochs_trained += 1
                    
                    ib.set_data(self.x_test, self.y_test)
                    
                    if self._is_multi_label is False:
                        y_hat = exec.eval_predict()                            
                    else:
                        y_hat = exec.eval_predict_multi()
                        
                    test_score = self.fn_test_score(self.y_test, np.array(y_hat))
                    test_log.append(test_score)                

                    self.fn_epoch_callback(epoch, train_acc, test_score)            
                    if test_score > best_test_score:
                        best_test_score = test_score
                        best_test_epoch = epoch
                        if self.load_best_state:
                            self._best_tm_state = self.tm.get_state_copy()                        
                    
                    
                    if test_score >= self.early_exit_acc:                                    
                        did_early_exit = True
                        break

                    progress_bar.set_description("Processing epoch {} of {}, train acc: {:.3f}, best test score: {:.3f} (epoch: {})".format(epoch+1,
                                                                                                                                self.n_epochs,
                                                                                                                                train_acc,
                                                                                                                                best_test_score,
                                                                                                                                best_test_epoch))
                    progress_bar.update(1)
                    
                    if epoch < (self.n_epochs - 1):                   
                        ib.set_data(self.x_train, self.y_train)

                    
            if self.load_best_state is True:
                self.tm.set_state(self._best_tm_state, copy=False) # copy False since we already created a copy in train().

        
        return {
            "best_test_score": best_test_score,
            "best_test_epoch": best_test_epoch,
            "n_epochs": n_epochs_trained,
            "train_log": train_log,
            "test_log": test_log,
            "did_early_exit": did_early_exit
        }

                
                
        
        
        
        
        
        
        
        
        
        
    

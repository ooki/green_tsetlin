import os
from typing import Optional, List, Union
from collections import defaultdict
from time import perf_counter

    
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold


#import green_tsetlin_core as gtc
from green_tsetlin import TsetlinMachine, DenseState, allocate_clause_blocks
from green_tsetlin.backend import impl as _backend_impl

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
                 copy_training_data:bool=True,
                 k_folds:int=0,
                 kfold_progress_bar:bool=False):

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

        self._best_tm_state : DenseState = None
        self.results : dict = None 

        self.kfold_progress_bar = kfold_progress_bar
        self.k_folds = k_folds

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

        self._cls_dense_ib = _backend_impl["dense_input"]
        self._cls_feedback_block = _backend_impl["feedback"]
        self._cls_feedback_block_uniform = _backend_impl["feedback_uniform"]
        self._cls_feedback_block_multi_label = _backend_impl["feedback_multi"]
        self._cls_exec_singlethread = _backend_impl["single_executor"]
        self._cls_exec_multithread = _backend_impl["thread_executor"]

        self._cls_sparse_ib = _backend_impl["sparse_input"]
        self._cls_input_block = None
        
        self.x_train = None


    def set_train_data(self, x_train:np.array, y_train:np.array):
        
        # raise error if data is sparse and cb is dense, and vice versa
        if isinstance(x_train, csr_matrix) and self.tm._backend_clause_block_cls == _backend_impl["cb"]:
            raise ValueError("x_train can not be csr_matrix when using dense tsetlin machine. To use this data with dense tsetlin machine, convert it to dense using .toarray() method.")

        if isinstance(x_train, np.ndarray) and self.tm._backend_clause_block_cls == _backend_impl["sparse_cb"]:
            raise ValueError("x_train can not be np.ndarray when using sparse tsetlin machine. To use this data with sparse tsetlin machine, convert it to sparse using scipy.sparse.csr_matrix().")

        
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
        
        if isinstance(x_test, csr_matrix) and self.tm._backend_clause_block_cls == _backend_impl["cb"]:
            raise ValueError("x_test can not be csr_matrix when using dense tsetlin machine. To use this data with dense tsetlin machine, convert it to dense using .toarray() method.")

        if isinstance(x_test, np.ndarray) and self.tm._backend_clause_block_cls == _backend_impl["sparse_cb"]:
            raise ValueError("x_test can not be np.ndarray when using sparse tsetlin machine. To use this data with sparse tsetlin machine, convert it to sparse using scipy.sparse.csr_matrix().")


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


    def set_validation_data(self, x_val:np.array, y_val:np.array):

        if isinstance(x_val, csr_matrix) and self.tm._backend_clause_block_cls == _backend_impl["cb"]:
            raise ValueError("x_val can not be csr_matrix when using dense tsetlin machine. To use this data with dense tsetlin machine, convert it to dense using .toarray() method.")

        if isinstance(x_val, np.ndarray) and self.tm._backend_clause_block_cls == _backend_impl["sparse_cb"]:
            raise ValueError("x_val can not be np.ndarray when using sparse tsetlin machine. To use this data with sparse tsetlin machine, convert it to sparse using scipy.sparse.csr_matrix().")

        y_val = np.atleast_1d(y_val)       

        if x_val is not None:
            x_val = x_val.copy()
            y_val = y_val.copy()

        if x_val.dtype != np.uint8:
            raise ValueError("Data x_test must be of type np.uint8, was: {}".format(x_val.dtype))
        
        if y_val.dtype != np.uint32:
            raise ValueError("Data y_test must be of type np.uint32, was: {}".format(y_val.dtype))
        
        if x_val.shape[1] != self.tm.n_literals:
            raise ValueError("Data x_test does not match in shape[1] (#literals) with n_literals : {} != {}".format(x_val.shape[1], self.tm.n_literals))
        
        self.x_val = x_val
        self.y_val = y_val 


    def _get_feedback_block(self, n_classes, threshold):
        if self.feedback_type == "focused_negative_sampling":
            if self.tm._is_multi_label is False:

                return self._cls_feedback_block(n_classes, threshold, self.seed)        
            else:
                # since we have actual classes as 2*classes since we use 0/1 per class
                return self._cls_feedback_block_multi_label(n_classes // 2, threshold, self.seed)
            

        elif self.feedback_type == "uniform":
            if self.tm._is_multi_label:
                raise NotImplementedError("Cannot use 'uniform' feedback for multi-label classification yet.")
            
            else:
                return self._cls_feedback_block_uniform(n_classes, threshold, self.seed)
            
        
        else:
            raise ValueError("Unknown feedback type: {}".format(self.feedback_type))
        
        
    def _calculate_blocks_for_tm(self):            
        return self.n_jobs


    def _select_backend_ib(self):

        if isinstance(self.x_train, csr_matrix) and isinstance(self.x_test, csr_matrix):
            self._cls_input_block = self._cls_sparse_ib
        
        elif isinstance(self.x_train, np.ndarray) and isinstance(self.x_test, np.ndarray):
            self._cls_input_block = self._cls_dense_ib
        
        else:
            raise ValueError("Train and test data must be of the same type. x_train type: {}, x_test type: {}".format(type(self.x_train), type(self.x_test)))


    def _train_inner(self):                
        
        if self.x_train is None:
            raise ValueError("Cannot train() without train data. Did you forget to set_train_data()?")

        self._select_backend_ib()
        input_block = self._cls_input_block(self.tm.n_literals)


        _flexible_set_data(input_block, self.x_train, self.y_train)

        feedback_block = self._get_feedback_block(self.tm.n_classes, self.tm.threshold)        
        
        if self.tm._clause_block_sizes is None:
            n_blocks = self._calculate_blocks_for_tm()
            self.tm.set_num_clause_blocks(n_blocks)
                
        cbs = self.tm.construct_clause_blocks()
        for cb in cbs:
            cb.set_feedback(feedback_block)     
            cb.set_input_block(input_block)
        
        if self.n_epochs < 1:            
            return
        
        with allocate_clause_blocks(cbs, seed=self.seed):    

            if self.tm._state is not None:
                self.tm._save_state_in_backend()

            if self.n_jobs == 1:
                exec = self._cls_exec_singlethread(input_block, cbs, feedback_block, 1, self.seed)
            else:
                exec = self._cls_exec_multithread(input_block, cbs, feedback_block, self.n_jobs, self.seed)
            

            # main loop
            n_epochs_trained = 0
            best_test_score = -1.0
            best_test_epoch = -1
            train_acc = -1.0
            
            train_time_of_epochs = []
            
            train_log = []
            test_log = []
            did_early_exit = False
            
            y_hat = np.empty_like(self.y_test)

            hide_progress_bar = self.progress_bar is False  
            with tqdm.tqdm(total=self.n_epochs, disable=hide_progress_bar) as progress_bar:
                progress_bar.set_description("Processing epoch 1 of {}, train acc: NA, best test score: NA".format(self.n_epochs))
        
                for epoch in range(self.n_epochs):         
                    
                    t0 = perf_counter()                                   
                    train_acc = exec.train_epoch()
                    t1 = perf_counter()

                    train_time_of_epochs.append(t1-t0)
                    train_log.append(train_acc)                
                    n_epochs_trained += 1
                    
                    _flexible_set_data(input_block, self.x_test, self.y_test)

                    if self.tm._is_multi_label is False:
                        exec.eval_predict(y_hat)                            
                    else:
                        exec.eval_predict_multi(y_hat)        
                            
                    test_score = self.fn_test_score(self.y_test, np.array(y_hat))
                    test_log.append(test_score)                
                    
                    self.fn_epoch_callback(epoch, train_acc, test_score)            
                    if test_score > best_test_score:
                        best_test_score = test_score
                        best_test_epoch = epoch
                        if self.load_best_state:
                            self._best_tm_state = self.tm._load_state_from_backend(only_return_copy=True)                                                

                    
                    
                    progress_bar.set_description("Processing epoch {} of {}, train acc: {:.3f}, best test score: {:.3f} (epoch: {})".format(epoch+1,
                                                                                                                                self.n_epochs,
                                                                                                                                train_acc,
                                                                                                                                best_test_score,
                                                                                                                                best_test_epoch))
                    progress_bar.update(1)
                    
                    if test_score >= self.early_exit_acc:                                    
                        did_early_exit = True
                        break
                    
                    if epoch < (self.n_epochs - 1):                   
                        _flexible_set_data(input_block, self.x_train, self.y_train)

                    
            if self.load_best_state is True:
                self.tm._state = self._best_tm_state
        
        r = {
            "train_time_of_epochs": train_time_of_epochs,
            "best_test_score": best_test_score,
            "best_test_epoch": best_test_epoch,
            "n_epochs": n_epochs_trained,
            "train_log": train_log,
            "test_log": test_log,
            "did_early_exit": did_early_exit
            }
        
        self.results = r
        
    
    def train(self):
        
        if self.k_folds > 1:

            x = np.vstack((self.x_train, self.x_test))
            y = np.concatenate((self.y_train, self.y_test))

            kf = StratifiedKFold(n_splits=self.k_folds, random_state=self.seed, shuffle=True)
            
            best_iter = -1
            best_score = -1

            with tqdm.tqdm(total=self.k_folds, disable=self.kfold_progress_bar is False) as progress_bar:
                
                progress_bar.set_description("Processing kfold 1 of {}, best test score: NA".format(self.k_folds))
                
                for i, (train_index, test_index) in enumerate(kf.split(x, y)):

                    x_train, x_test = x[train_index], x[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    x_train = np.ascontiguousarray(x_train, dtype=np.uint8)
                    x_test = np.ascontiguousarray(x_test, dtype=np.uint8)
                    y_train = np.ascontiguousarray(y_train, dtype=np.uint32)
                    y_test = np.ascontiguousarray(y_test, dtype=np.uint32)

                    self.set_train_data(x_train, y_train)
                    self.set_test_data(x_test, y_test)

                    self._train_inner()

                    if self.results["best_test_score"] > best_score:
                        
                        best_score = self.results["best_test_score"]
                        best_iter = i

                    progress_bar.set_description("Processing kfold {} of {}, best test score: {:.3f} (iter: {})".format(i+1, self.k_folds, best_score, best_iter))
                    progress_bar.update(1)
            
            r = {"best_test_score": best_score, "k_folds" : self.k_folds}

            self.results = r
            return self.results

        else:
            self._train_inner()
            return self.results


def _flexible_set_data(ib, x, y):
    if isinstance(x, csr_matrix):
        ib.set_data(x.indices, x.indptr, y)
    elif isinstance(x, np.ndarray):
        ib.set_data(x, y)
    else:
        raise ValueError("Cannot set input data with type: {}".format(type(x)))
        
        
    

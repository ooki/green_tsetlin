
import os
from typing import Optional, List, Union
from collections import defaultdict

    
import numpy as np
from sklearn.metrics import accuracy_score
import tqdm


import green_tsetlin_core as gtc
from green_tsetlin import TsetlinMachine


def empty_epoch_callback(epoch, train_acc, test_score):
    pass

class Trainer:
    """Trainer for a Tsetlin Machine. 
    
    
    This class train a coalesced tsetlin machine in a supervised manner.
    
    
    Parameters
    ----------
    threshold : int, bound for number of votes (-threshold < votes < threshold).
    
    feedback_type: str, {"focused_negative_sampling"}, default="focused_negative_sampling"
                Select how the feedback to the TM should be given.        
            
            - "focused_negative_sampling", proportionally select the most 'wrong' negative class and update it.
        
        
    n_jobs: int, number of threads to run during training. default=1    
        The Trainer will split the TM into blocks so
        that each block gets it own thread. The performance may degrade if the number of threads gets too high compared to the number of clauses per block.  
        If multiple TMs are used the blocks are diveded as follows: first all tm gets 1 block, then the highest clause count gets an additional
        block, while halving the number of clauses in that block. 
        For instance the following number of clauses [10, 30, 100] and 6 jobs, will result in [1,2,3] blocks.
        
        If set to -1, all CPUs are used (from os.cpu_count()). For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        For example with n_jobs=-2, all CPUs but one are used.)
    
    n_epochs: int, number of epochs to train for. default=20
    
    seed: int, seed for the random generators. 0 gives a random seed. default=0
    
    early_exit_acc: float, if the score (e.g. accuracy) gets larger than this, perform a early exit. default=1.0
    
    load_best_state: bool, if true overwrite the state of each TM with the best state found during training. default=True
    
    fn_epoch_callback: callback on epoch end - can for instance be used to track progress. default=empty_epoch_callback
                callback signature: callback_on_epoch(epoch, train_acc, test_score)
                
    
    fn_test_score: Callback to generate a test score for the epoch. default="accuracy"
                    If "accuracy" then acuraccy will be calculated. 
                    Can also be a callback with signature:  callback( y_true : np.array, y_pred : np.array )
                    
    progress_bar: bool, show progress bar (tqdm). default=True
    
    """
    def __init__(self, threshold: int,
                 feedback_type="focused_negative_sampling",
                 n_jobs:int=1,
                 n_epochs: int = 20,
                 seed:int=0,
                 early_exit_acc:float=1.0,
                 load_best_state:bool=True,
                 fn_epoch_callback=empty_epoch_callback,
                 fn_test_score="accuracy",
                 progress_bar=True):                

        self.threshold = threshold        
        self.feedback_type = feedback_type
        self.seed = seed
        self.n_epochs = n_epochs
        self.load_best_state = load_best_state
        self.early_exit_acc = early_exit_acc
        self.progress_bar = progress_bar
        
        self.fn_epoch_callback = fn_epoch_callback        
        
        if n_jobs < 0:
            n_cpus = os.cpu_count()                        
            self.n_jobs = n_cpus + 1 + n_jobs
            
        elif n_jobs == 0:
            self.n_jobs = 1
            
        else:
            self.n_jobs = n_jobs
        

        
        if fn_test_score == "accuracy":
            self.fn_test_score = accuracy_score
        
        else:
            self.fn_test_score = fn_test_score
            

        if self.threshold < 1:
            raise ValueError("threshold cannot be less than 1, is: {}".format(self.threshold))
        
        self._is_multi_label : bool = None
        self.n_blocks_used : int = 0
        self.best_state : list = None
        
    def _get_feedback_block(self, n_classes, threshold):
        if self.feedback_type == "focused_negative_sampling":
            if self._is_multi_label is False:
                return gtc.FeedbackBlock(n_classes, threshold, self.seed)
            else:
                return gtc.FeedbackBlockMultiLabel(n_classes // 2, threshold, self.seed) # since we have actual classes as 2*classes since we use 0/1 per class
            
        
        else:
            raise ValueError("Unknown feedback type: {}".format(self.feedback_type))
        
    
    def _save_state(self, tms):
        for tm in tms:            
            tm.store_state()


    def _calculate_blocks_per_tm(self, tms:List[TsetlinMachine]) -> List[int]:
        
        if len(tms) < 2:
            return [self.n_jobs]
        
        clauses = np.array([tm.n_clauses for tm in tms], dtype=float)
        blocks = [1] * len(tms)
        
        while sum(blocks) < self.n_jobs:
            idx_most_clauses = np.argmax(clauses)
            clauses[idx_most_clauses] *= 0.5
            blocks[idx_most_clauses] += 1
            
        return blocks
            
        
        
        
        


    def train(self, tms: Union[TsetlinMachine, List[TsetlinMachine]]) -> dict:
        """ Perform the training of a TM with the paramters set in the constructor.

        Args:
            tms (Union[TsetlinMachine, List[TsetlinMachine]]): TM's to train, can either be a single TM or a list of TM's.
            All TM's must have the same number of classes.

        Returns:
            dict: {
                "best_test_score": best_test_score,
                "best_test_epoch": best_test_epoch,
                n_epochs": n_epochs_trained,
                "train_log": train_log,
                "test_log": test_log,
                "did_early_exit": did_early_exit
        }
        """
        if isinstance(tms, TsetlinMachine):
            tms = [tms]

        # check that we have the same number of classes in all tsetlin machines
        if len(set([tm.n_classes for tm in tms])) != 1:
            raise ValueError("All TsetlinMachines in a single trainer instance must have the same number of classes.")

        self._is_multi_label = tms[0]._is_multi_label
        if all([tm._is_multi_label==tms[0]._is_multi_label for tm in tms]) is False:
            raise ValueError("Cannot mix single label and multi label tm's in the same Trainer")
        
        n_classes = tms[0].n_classes
        if all([tm.n_classes==n_classes for tm in tms]) is False:
            raise ValueError("All Tsetlin Machines must have same number of classes: {}".format(n_classes))
                

        label_tm = None
        ibs = []                        
        for tm in tms:
            ib = gtc.DenseInputBlock(tm.n_literals)
            ib.set_data(tm._train_x, tm._train_y)
            ibs.append(ib)

            if tm._train_y.size > 0:
                if label_tm is not None:
                    raise ValueError("Multiple TM has labels set, only one set of labels should be used.")
                
                label_tm = tm

        if label_tm is None:
            raise ValueError("No TM has labels, please use set_train_data() to spesify labels.")
        
             
        feedback_block = self._get_feedback_block(n_classes, self.threshold)
        

           
        cb_seed = self.seed
        all_cbs = []
        block_allocation = self._calculate_blocks_per_tm(tms)
        
        for k, (ib, tm) in enumerate(zip(ibs, tms)):
            
            n_blocks = block_allocation[k]
            cbs = tm.construct_clause_blocks(n_blocks)
            
            for cb in cbs:            
                cb.set_feedback(feedback_block)
                cb.set_input_block(ib)
                
                cb.initialize(cb_seed)                
                cb_seed += 1
                
            all_cbs.extend(cbs)
        
        self.n_blocks_used = len(all_cbs)

        # if no epochs, clean and exit
        if self.n_epochs < 1:            
            for cb in all_cbs:
                cb.cleanup()
            return
        
            
        if self.n_jobs == 1:
            exec = gtc.SingleThreadExecutor(ibs, all_cbs, feedback_block, self.seed)
        else:
            exec = gtc.MultiThreadExecutor(ibs, all_cbs, feedback_block, self.n_jobs, self.seed)
        
        
        n_epochs_trained = 0
        best_test_score = -1.0
        best_test_epoch = -1
        train_acc = -1.0
        
        train_log = []
        test_log = []
        did_early_exit = False

        hide_progress_bar = self.progress_bar is False  
        with tqdm.tqdm(total=self.n_epochs, disable=hide_progress_bar) as progress_bar:
            progress_bar.set_description("Processing epoch 1 of {}, train acc: N\A, best test score: N\A".format(self.n_epochs))
    
            for epoch in range(self.n_epochs):                                            
                train_acc = exec.train_epoch()
                train_log.append(train_acc)                
                n_epochs_trained += 1

                # set data : test
                for ib, tm in zip(ibs, tms):
                    ib.set_data(tm._test_x, tm._test_y)
                                
                if self._is_multi_label is False:
                    y_hat = exec.eval_predict()                            
                else:
                    y_hat = exec.eval_predict_multi()
                    
                test_score = self.fn_test_score(label_tm._test_y, np.array(y_hat))
                
                test_log.append(test_score)

                self.fn_epoch_callback(epoch, train_acc, test_score)            
                if test_score > best_test_score:
                    best_test_score = test_score
                    best_test_epoch = epoch
                    if self.load_best_state:
                        self._save_state(tms)
                
                
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
                    for ib, tm in zip(ibs, tms):
                        ib.set_data(tm._train_x, tm._train_y)
            
        # save last state     
        if self.load_best_state is False:
            self._save_state(tms)
                    
        for cb in all_cbs:
            cb.cleanup()
        
        return {
            "best_test_score": best_test_score,
            "best_test_epoch": best_test_epoch,
            "n_epochs": n_epochs_trained,
            "train_log": train_log,
            "test_log": test_log,
            "did_early_exit": did_early_exit
        }






           


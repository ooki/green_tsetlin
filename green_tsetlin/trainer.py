

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
        self.n_jobs = n_jobs
        self.n_epochs = n_epochs
        self.load_best_state = load_best_state
        self.early_exit_acc = early_exit_acc
        self.progress_bar = progress_bar
        
        self.fn_epoch_callback = fn_epoch_callback
        
        if fn_test_score == "accuracy":
            self.fn_test_score = accuracy_score
        
        else:
            self.fn_test_score = fn_test_score
            

        if self.threshold < 1:
            raise ValueError("threshold cannot be less than 1, is: {}".format(self.threshold))
        

        self.best_state : list = None
        
    def _get_feedback_block(self, n_classes, threshold):
        if self.feedback_type == "focused_negative_sampling":
            return gtc.FeedbackBlock(n_classes, threshold)
        
        else:
            raise ValueError("Unknown feedback type: {}".format(self.feedback_type))
        
    
    def _save_state(self, tms):
        for tm in tms:            
            tm.store_state()



    def train(self, tms: Union[TsetlinMachine, List[TsetlinMachine]]) -> dict:
        if isinstance(tms, TsetlinMachine):
            tms = [tms]

        # check that we have the same number of classes in all tsetlin machines
        if len(set([tm.n_classes for tm in tms])) != 1:
            raise ValueError("All TsetlinMachines in a single trainer instance must have the same number of classes.")

        n_classes = tms[0].n_classes
        feedback_block = self._get_feedback_block(n_classes, self.threshold)

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

           
        cb_seed = self.seed
        all_cbs = []
        for ib, tm in zip(ibs, tms):
            
            # TODO: fix so the number of blocks/jobs is based on all #clauses 
            n_blocks = max(self.n_jobs, 1)
            cbs = tm.construct_clause_blocks(n_blocks)
            
            for cb in cbs:            
                cb.set_feedback(feedback_block)
                cb.set_input_block(ib)
                
                cb.initialize(cb_seed)                
                cb_seed += 1
                
            all_cbs.extend(cbs)
            

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
                                
                y_hat = exec.eval_predict()                            
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






           


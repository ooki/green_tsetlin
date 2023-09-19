
from typing import Optional, List, Union
import uuid

import numpy as np
import scipy.sparse as sparse

import green_tsetlin_core as gtc



# set the default core TM type based on whats available in green_tsetlin_core (binary)
_tm_cls = gtc.ClauseBlockNV
try:
    _tm_cls = gtc.ClauseBlockNeon
except AttributeError:
    pass

try:
    _tm_cls = gtc.ClauseBlockAVX2
except AttributeError:
    pass



class TsetlinMachine:
    """ Represent a Tsetlin Machine that can be trained with a Trainer instance.
    
    The actual allocation is not done until the train() method is called from a trainer.
    
    
    Parameters
    ----------
    n_literals: int, number of literals
    n_clauses: int, number of clauses to use
    n_classes: int, number of classes to select from
    s: float, spesifisity, must be more than 1.0. 
    n_literal_budget: int, soft cap on how many literals that can be active in a clause. (-1 to turn off), default=-1
    multi_label: bool, Run on multi label or single label, default=False
    
    """
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int,  s:float, n_literal_budget: Optional[int] = -1, multi_label:bool=False, positive_budget:bool=False):
        
        self.name = hash(uuid.uuid4().hex)
        
        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        self.s = s
        if self.s < 1.0:
            raise ValueError("Cannot set s paramter to less than 1.0, (set to: {})".format(s))

        self._is_multi_label:bool = multi_label
        if self._is_multi_label:
            self.n_classes *= 2 # since each class can now be both ON and OFF (each has its own TM weight)


        self.n_literals_budget = n_literal_budget
        if self.n_literals_budget < 1:
            self.n_literals_budget = self.n_literals             
        
        self.positive_budget = positive_budget
        self._tm_cls = _tm_cls

        if self.positive_budget is True:            
            if self._tm_cls == gtc.ClauseBlockAVX2:
                self._tm_cls = gtc.ClauseBlockAVX2PB
            
            elif self._tm_cls == gtc.ClauseBlockNV:
                self._tm_cls = gtc.ClauseBlockNVPB

            else:
                raise NotImplementedError("Positive Budget is not implemented for: {}".format(self._tm_cls))
            


        self._train_x : np.array = None
        self._train_y : np.array = None
        self._test_x : np.array = None
        self._test_y : np.array = None

        self._cbs = None
        self._state = None
        
        
    def get_state(self):
        """ Get the state of the TM after training.
            Note: does NOT return a copy
        
        Raises:
            ValueError: If the TM is not trained.

        Returns:
            dict: {"w": weights, "c": clauses}
        """
        if self._state is None:
            raise ValueError("Cannot get the state of a non-trained Tsetlin Machine")
        
        return self._state
    
    def save_state_to_file(self, file_path:str) -> None:
        """Saves the internal state to a compressed .npz file for easy loading
        Does not use pickle, and should be safe to distribute.
        See: https://numpy.org/doc/stable/reference/security.html

        Args:
            file_path (str): Out file, will throw if it fails to open file.
        """
        if self._state is None:
            raise ValueError("Cannot save a empty Tsetlin Machine without a stored state. Is the Tsetlin Machine trained?")

        np.savez_compressed(file_path, w=self._state["w"], c=self._state["c"])

    def load_state_from_file(self, file_path:str) -> None:
        """Load the state of a tm from a .npz file. Will not recover hyper-parameters.
        Does not use pickle, and should be safe to distribute.
        See: https://numpy.org/doc/stable/reference/security.html

        Args:
            file_path (str): file to load.
        """
        o = np.load(file_path)

        d = {"w": o["w"], "c": o["c"]}
        self.set_state(d)


        
    def __hash__(self) -> int:
        return self.name

    def set_train_data(self, x: np.array, y: Optional[np.array] = None) -> None:
        """ Set the training data for this Tsetlin Machine.
        
        Args:
            x np.array: input data must be np.uint8, should be of size: (n_examples, n_literals)
            
            y np.array]: labels, must be np.int32, should be of size (n_examples). Default: None
                         Note that if multiple TM's are trained with a single trainer only 1 should have
                         the y (labels) set. The rest should be None. 
        """

        if x.dtype != np.uint8:
            raise ValueError("Train data x must be of type np.uint8, was: {}".format(x.dtype))                
                   
        if x.shape[1] != self.n_literals:
            raise ValueError("Train data x does not match in shape[1] (#literals) with n_literals : {} != {}".format(x.shape[1], self.n_literals))
        
        if sparse.issparse(x):
            raise ValueError("TsetlinMachine does not accept sparse matrices (yet).")
        
        self._train_x = x

        if y is None:
            self._train_y = np.empty(shape=(0,0), dtype=np.int32)
        
        else:
            if y.dtype != np.uint32:
                raise ValueError("Train data y must be of type np.uint32, was: {}".format(y.dtype))
        
            if x.shape[0] != y.shape[0]:
                raise ValueError("Train data x/y does not match in shape[0] (example index) : {} != {}".format(x.shape[0], y.shape[0]))
                        
            if self._is_multi_label is False and y.ndim > 1:
                raise ValueError("TsetlinMachine is flagged as single label - but multi label y is set (train)")
            
            n_expected_classes = self.n_classes // 2
            if self._is_multi_label is True and y.shape[1] != n_expected_classes:
                raise ValueError("Multi label TsetlinMachine need 0/1 encoded multi label, got {} (expected: {}) (train)".format(
                    y.shape[1], n_expected_classes))
                            
            self._train_y = y            
        
    def set_test_data(self, x: np.array, y: Optional[np.array] = None) -> None:
        """ Set the test data for this Tsetlin Machine.
        
        Args:
            x np.array: input data must be np.uint8, should be of size: (n_examples, n_literals)
            
            y np.array]: labels, must be np.uint32, should be of size (n_examples). Default: None
                         Note that if multiple TM's are trained with a single trainer only 1 should have
                         the y (labels) set. The rest should be None. 
        """
        
        if x.dtype != np.uint8:
            raise ValueError("Test data x must be of type np.uint8, was: {}".format(x.dtype))                

        if x.shape[1] != self.n_literals:
            raise ValueError("Test data x does not match in shape[1] (#literals) with n_literals : {} != {}".format(x.shape[1], self.n_literals))
        
        if sparse.issparse(x):
            raise ValueError("TsetlinMachine does not accept sparse matrices (yet).")
        
        self._test_x = x        
        
        if y is None:
            self._test_y = np.empty(shape=(0,0), dtype=np.uint32)
            
        else:
            if y.dtype != np.uint32:
                    raise ValueError("Test data y must be of type np.uint32, was: {}".format(y.dtype))
            
            if x.shape[0] != y.shape[0]:
                raise ValueError("Test data x/y does not match in shape[0] (example index) : {} != {}".format(x.shape[0], y.shape[0]))
            
            if self._is_multi_label is False and y.ndim > 1:
                raise ValueError("TsetlinMachine is flagged as single label - but multi label y is set (test)")
            
                
            self._test_y = y                   
        
    def construct_clause_blocks(self, n_blocks:int=1):
        """construct_clause_blocks 

        Args:
            n_blocks: number of blocks to create. Defaults to 1.
            
        Returns a list of clause blocks (of type given by self._tm_cls)
        """
        n_clause_per_block = self.n_clauses // n_blocks
        n_add = self.n_clauses % n_blocks
        
        assert (n_clause_per_block * n_blocks) + n_add == self.n_clauses
        
        self._cbs = []
        for k in range(n_blocks):
            if k > 0:
                n_add = 0
            
            cb = self._tm_cls(self.n_literals, n_clause_per_block + n_add, self.n_classes)
            cb.set_s(self.s)
            cb.set_literal_budget(self.n_literals_budget)
            
            self._cbs.append(cb)
        
        return self._cbs
    
    def store_state(self):  
        if self._state is None: # allocate state
            self._state = {}
            self._state["c"] = np.empty(shape=(self.n_clauses, self.n_literals*2), dtype=np.int8)
            self._state["w"] = np.empty(shape=(self.n_clauses, self.n_classes), dtype=np.int16)
        
        clause_offset = 0 
        for cb in self._cbs:
            cb.get_clause_state(self._state["c"], clause_offset)
            cb.get_clause_weights(self._state["w"], clause_offset)
            clause_offset += cb.get_number_of_clauses()
    
    
    def set_state(self, state):
        """set_state of the TM (clauses and weights).

        Args:
            state: dict {
                "c": np array: dtype:int8, size: (n_clauses, n_literals*2)
                "w": np array: dtype:int16, size: (n_clauses, n_classes                
                }
        """
        c = state["c"]
        w = state["w"]
        
        if c.dtype != np.int8:
            raise ValueError("Clause state much be np.int8 is {}".format(c.dtype))
        
        if c.shape[0] != self.n_clauses or c.shape[1] != (self.n_literals*2):
            raise ValueError("Clause State array is of wrong shape ({}) should be {}".format(c.shape, (self.n_clauses, self.n_literals*2)))
        
        if w.dtype != np.int16:
            raise ValueError("Clause Weights much be np.int16 is {}".format(w.dtype))
        
        if w.shape[0] != self.n_clauses or w.shape[1] != self.n_classes:
            raise ValueError("Clause Weights array is of wrong shape {} should be {}".format(w.shape, (self.n_clauses, self.n_classes)))
       
            
        clause_offset = 0 
        for cb in self._cbs:
            cb.set_clause_state(c, clause_offset)
            cb.set_clause_weights(w, clause_offset)
            clause_offset += cb.get_number_of_clauses()
            
        
        


if __name__ == "__main__":

    tm = TsetlinMachine(4, 2, 2, 1.5, 2)






















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
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int,  s:float, n_literal_budget: Optional[int] = -1):
        
        self.name = hash(uuid.uuid4().hex)
        
        self.n_literals = n_literals
        self.n_clauses = n_clauses 
        self.n_classes = n_classes

        self.s = s
        if self.s < 1.0:
            raise ValueError("Cannot set s paramter to less than 1.0, (set to: {})".format(s))

        self.n_literals_budget = n_literal_budget
        if self.n_literals_budget < 1:
            self.n_literals_budget = self.n_literals             
        
        self._tm_cls = _tm_cls

        self._train_x : np.array = None
        self._train_y : np.array = None
        self._test_x : np.array = None
        self._test_y : np.array = None

        self._cbs = None
        self._state = None
        
    def get_state(self):
        if self._state is None:
            raise ValueError("Cannot get the state of a non-trained Tsetlin Machine")
        
        return self._state
        
    def __hash__(self) -> int:
        return self.name

    def set_train_data(self, x: np.array, y: Optional[np.array] = None) -> None:

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
            
            self._train_y = y            
        
    def set_test_data(self, x: np.array, y: Optional[np.array] = None) -> None:
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
            
            self._test_y = y                   
        
    def construct_clause_blocks(self, n_blocks:int=1):
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
        c = state["c"]
        w = state["w"]
        
        clause_offset = 0 
        for cb in self._cbs:
            cb.set_clause_state(self._state["c"], clause_offset)
            cb.set_clause_weights(self._state["w"], clause_offset)
            clause_offset += cb.get_number_of_clauses()
            
        
        


if __name__ == "__main__":

    tm = TsetlinMachine(4, 2, 2, 1.5, 2)





















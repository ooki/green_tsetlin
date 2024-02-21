



import numpy as np
import green_tsetlin.py_gtc as py_gtc

class TsetlinStateStorage:
    """
    A storage object for a Tsetlin Machine state.
    w : is the class Weights
    c : is the Clauses
    """
    def __init__(self, w:np.array, c:np.array):
        self.w = w
        self.c = c
    


class TsetlinMachine:
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, s : float, threshold: int, multi_label:bool=False):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self.cbs : list = None

        self.s = s
        if s < 1.0:
            raise ValueError("Cannot have a s value under 1.0 (currently: {}))".format(s))
        
        self.threshold = threshold
        if threshold < 1:
            raise ValueError("Cannot have a threshold  value under 1 (currently: {}))".format(threshold))

        self._is_multi_label:bool = multi_label
        if self._is_multi_label:
            self.n_classes *= 2 # since each class can now be both ON and OFF (each has its own TM weight)

        
        self._backend = py_gtc.ClauseBlock


    def get_state_copy(self) -> TsetlinStateStorage:
        """ Return a copy of the state        
        """
        raise NotImplementedError("Not Impl.")
    
    def set_state(self, state:TsetlinStateStorage, copy:bool=True):
        """ Set the internal state

        Args:
            copy (bool, optional): If True, set a copy of the state. Else directly set the state (shared-memory).
        """
        raise NotImplementedError("Not Impl.")
    

    def construct_clause_blocks(self, n_blocks):
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
        




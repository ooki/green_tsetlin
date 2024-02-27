
from typing import Union, Optional
import itertools
from contextlib import contextmanager
import copy


import numpy as np
import green_tsetlin.py_gtc as py_gtc

class TMState:
    """
    A storage object for a Tsetlin Machine state.
    w : is the class Weights
    c : is the Clauses

    Will not create any copy of the underlying array, but will create a reference.
    So the user is responsible for not changing the state after it has been loaded OR provide a copy.
    """
    def __init__(self, n_literals:Optional[int]=None, n_clauses:Optional[int]=None, n_classes:Optional[int]=None):
        if n_literals is not None:
            self.w = np.zeros(shape=(n_clauses, n_classes), dtype=np.int16)
            self.c = np.zeros(shape=(n_clauses, n_literals*2), dtype=np.int8)
        else:
            self.w:np.array = None
            self.c:np.array = None        


    def copy(self) -> "TMState":
        tm = TMState()
        tm.w = self.w.copy()
        tm.c = self.c.copy()
        return tm

    @staticmethod
    def load_from_file(file_path) -> "TMState":
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        d = np.load(file_path)
        tms = TMState()

        if tms["w"].shape[0] != d["c"].shape[0]:
            raise ValueError("Cannot load state. w and c must have the same number of clauses")        

        if tms["w"].dtype != np.int16:
            raise ValueError("Clause Weights much be np.int16 is {}".format(tms["w"].dtype))
        
        if tms["c"].dtype != np.int8:
            raise ValueError("Clause State much be np.int8 is {}".format(tms["c"].dtype))

        tms.w = d["w"]
        tms.c = d["c"]
        return tms
    

    def save_to_file(self, file_path) -> None:
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        
        np.savez(file_path, w=self.w, c=self.c)






class TsetlinMachine:
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, s : Union[float, list], threshold: int, multi_label:bool=False):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes
        self._cbs : list = None
        self._state : TMState = None
        self.n_literals_budget = 32_700 # high value, should really be set.

        if isinstance(s, float):
            s = [s]
        
        self.s = s
        for s_i in self.s:
            if s_i < 1.0:
                raise ValueError("Cannot have a s value under 1.0 (currently: {}))".format(s_i))
        
        self.threshold = threshold
        if threshold < 1:
            raise ValueError("Cannot have a threshold  value under 1 (currently: {}))".format(threshold))

        self._is_multi_label:bool = multi_label
        if self._is_multi_label:
            self.n_classes *= 2 # since each class can now be both ON and OFF (each has its own TM weight)

        
        self._backend_clause_block_cls = py_gtc.ClauseBlock


    def _load_state_from_backend(self):
        """ Collects the state from the backend and store it in the state_ variable.         
        """
        if self._state is None: # allocate state
            self._state = TMState(n_literals=self.n_literals, n_clauses=self.n_clauses, n_classes=self.n_classes)
        
        clause_offset = 0 
        for cb in self._cbs:
            cb.get_clause_state(self._state.c, clause_offset)
            cb.get_clause_weights(self._state.w, clause_offset)
            clause_offset += cb.get_number_of_clauses()

    
    def _save_state_in_backend(self):
        """ Set the internal state
        """
        if self._cbs is None:
            raise ValueError("There is no clauseblocks in the backend, cannot save state")
        
        if self._state is None:
            raise ValueError("Cannot save a empty Tsetlin Machine into the backend. Please load a state first.")
        
        clause_offset = 0 
        for cb in self._cbs:
            cb.set_clause_state(self._state.c, clause_offset)
            cb.set_clause_weights(self._state.w, clause_offset)
            clause_offset += cb.get_number_of_clauses()


    def load_state(self, path_or_state: Union[str, TMState]):
        if isinstance(path_or_state, str):
            self._state = TMState.load_from_file(path_or_state)
        else:
            self._state = path_or_state


    def save_state(self, path:str):
        if self._state is None:

            if self._cbs is not None:
                self._load_state_from_backend()
            else:
                raise ValueError("Cannot save a empty Tsetlin Machine without a stored state. Is the Tsetlin Machine trained?")
                    
        self._state.save_to_file(path)

    
    

    def construct_clause_blocks(self, n_blocks:int) -> list:
        """_summary_

        Args:
            n_blocks (int): The number of blocks to create. If the reminder is not zero it will be added to the last block.

        Returns:
            list: A list of clause_blocks of type self._backend_clause_block_cls, the list is a copy while the members are shared.
        """
        n_clause_per_block = self.n_clauses // n_blocks
        n_add = self.n_clauses % n_blocks
        
        assert (n_clause_per_block * n_blocks) + n_add == self.n_clauses
        
        self._cbs = []
        for s_k, k in zip(itertools.cycle(self.s), range(n_blocks)):
            if k > 0:
                n_add = 0
            
            cb = self._backend_clause_block_cls(self.n_literals, n_clause_per_block + n_add, self.n_classes)
            cb.set_s(s_k)
            cb.set_literal_budget(self.n_literals_budget)
            
            self._cbs.append(cb)
        
        return copy.copy(self._cbs)
        

@contextmanager
def allocate_clause_blocks(cbs_or_tm: Union[list, TsetlinMachine] , seed: int):
    if isinstance(cbs_or_tm, TsetlinMachine):
        cbs = cbs_or_tm._cbs
        if cbs is None:
            raise ValueError("The Tsetlin Machine has no clause blocks. Did you call constrcut_clause_blocks() first?")
    else:
        cbs = cbs_or_tm


    running_seed = seed
    for cb in cbs:
        cb.initialize(seed=running_seed)
        running_seed += 1

    yield

    for cb in cbs:
        cb.cleanup()

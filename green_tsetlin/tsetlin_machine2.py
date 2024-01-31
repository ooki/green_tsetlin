




import numpy as np


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
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, multi_label:bool=False):

        self.n_literals = n_literals
        self.n_clauses = self.n_clauses
        self.n_classes = self.n_classes

        
        self._is_multi_label:bool = multi_label
        if self._is_multi_label:
            self.n_classes *= 2 # since each class can now be both ON and OFF (each has its own TM weight)

        _tm_backend = None  # TODO: insert correct tm backend here
        self._backend = _tm_backend


    def get_state(self, copy:bool=True) -> TsetlinStateStorage:
        """ Return a copy of the state

        Args:
            copy (bool, optional): If True return a copy of the state, else return the internal memory. Defaults to True.
        """

        raise NotImplementedError("Not Impl.")
    
    def set_state(self, state:TsetlinStateStorage, copy:bool=True):
        """ Set the internal state

        Args:
            copy (bool, optional): If True, set a copy of the state. Else directly set the state (shared-memory).
        """
        raise NotImplementedError("Not Impl.")
    





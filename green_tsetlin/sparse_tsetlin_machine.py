
from typing import Union, Optional, List
import itertools
from contextlib import contextmanager
import copy

import numpy as np

import green_tsetlin as gt
from green_tsetlin.backend import impl as _backend_impl
import pickle

from green_tsetlin import TsetlinMachine


class SparseState:
    """
    A storage object for a Tsetlin Machine state.
    w : is the class Weights
    c_data : is the Clauses data
    c_indices : is the Clauses indices
    c_indptr : is the Clauses indptr
    AL : is the Active Literals

    c_data, c_indices and c_indptr are the data, indices and indptr of a CSR matrix.

    Will not create any copy of the underlying array, but will create a reference.
     So the user is responsible for not changing the state after it has been loaded OR provide a copy.
    """

    def __init__(self, n_literals:Optional[int] = None, n_clauses:Optional[int] = None, n_classes:Optional[int] = None):
        self.n_literals = n_literals
        if n_literals is not None:
            self.w = np.zeros(shape=(n_clauses, n_classes), dtype=np.int16)

            # these are returned not filled like for dense_state, think not need to pre allocate space
            self.c_data = []
            self.c_indices = []
            self.c_indptr = []
            self.AL = []

        else:
            self.w:np.array = None
            self.c_data:np.array = None
            self.c_indices:np.array = None
            self.c_indptr:np.array = None
            self.AL:np.array = None


    def copy(self) -> "SparseState":
        tm = SparseState()
        tm.w = self.w.copy()
        tm.c_data = self.c_data.copy()
        tm.c_indices = self.c_indices.copy()
        tm.c_indptr = self.c_indptr.copy()
        tm.AL = self.AL.copy()
        return tm

    @staticmethod
    def load_from_file(file_path) -> "SparseState":
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        d = np.load(file_path, allow_pickle=True)
        tms = SparseState()


        
        # temp_c_data = d["c_data"].astype(np.int8)
        # temp_c_indices = d["c_indices"].astype(np.uint32)
        # temp_c_indptr = d["c_indptr"].astype(np.uint32)
        # temp_AL = d["AL"].astype(np.uint32)

        # if d["w"].shape[0]*2 != d["c_indptr"][0].shape[0]-1:
        #     raise ValueError("Cannot load state. w and c must have the same number of clauses, w_size: {}, c_size: {}".format(d["w"].shape[0], d["c_indptr"][0].shape[0]-1)) 

        if d["w"].dtype != np.int16:
            raise ValueError("Clause Weights much be np.int16 is {}".format(d["w"].dtype))


        tms.w = d["w"]
        tms.c_data = d["c_data"].tolist()
        tms.c_indices = d["c_indices"].tolist()
        tms.c_indptr = d["c_indptr"].tolist()
        tms.AL = d["AL"].tolist()
        tms.n_literals = d["n_literals"]
        return tms

    def save_to_file(self, file_path) -> None:
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        
        np.savez(file_path, w=self.w, c_data=np.array(self.c_data, dtype=object), c_indices=np.array(self.c_indices, dtype=object), c_indptr=np.array(self.c_indptr, dtype=object), AL=np.array(self.AL, dtype=object), n_literals=self.n_literals)
        
        

class SparseTsetlinMachine(TsetlinMachine):
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, s : Union[float, list], threshold: int,
                 literal_budget:Optional[int]=None, boost_true_positives: bool = False, dynamic_AL: bool = True, multi_label:bool=False):
        
        super().__init__(n_literals, n_clauses, n_classes, s, threshold, literal_budget, boost_true_positives, multi_label)
        
        
        self.clause_size = np.ceil(np.sqrt(n_literals)).astype(int)
        self.active_literals_size = np.ceil(np.sqrt(n_literals)).astype(int)
        self.lower_ta_threshold = -40
        self.dynamic_AL = dynamic_AL
        
        self._backend_clause_block_cls = _backend_impl["sparse_cb"]


    def _load_state_from_backend(self, only_return_copy:bool=False):
        """ Collects the state from the backend.
        if only_return_copy is True => just return the state, else set it to state_ in the TM                        
        """
        
        state = self._state
        if only_return_copy:
            state = None

        # if state is None: # allocate state
        # need do this to clear the state, as the values are not replaced like in dense
        state = SparseState(n_literals=self.n_literals, n_clauses=self.n_clauses, n_classes=self.n_classes)


        
        clause_offset = 0 
        for cb in self._cbs:
            _c_state = cb.get_clause_state_sparse()
            state.c_data.append(_c_state[0])
            state.c_indices.append(_c_state[1])
            state.c_indptr.append(_c_state[2])
            state.AL.append(cb.get_active_literals())
            cb.get_clause_weights(state.w, clause_offset)
            clause_offset += cb.get_number_of_clauses()

        if only_return_copy:
            return state
        else:
            self._state = state
    
    def _save_state_in_backend(self):
        """ Set the internal state
        """
        if self._cbs is None:
            raise ValueError("There is no clauseblocks in the backend, cannot save state")
        
        if self._state is None:
            raise ValueError("Cannot save a empty Tsetlin Machine into the backend. Please load a state first.")
        
        clause_offset = 0 
        for index, cb in enumerate(self._cbs):
            cb.set_clause_state_sparse(self._state.c_data[index], self._state.c_indices[index], self._state.c_indptr[index])
            cb.set_clause_weights(self._state.w, clause_offset)
            cb.set_active_literals(self._state.AL[index])
            clause_offset += cb.get_number_of_clauses()


    def load_state(self, path_or_state: Union[str, SparseState]):
        """
        Load the state from the given path or state object.

        Parameters:
            path_or_state (Union[str, SparseState]): The path to the state file or the SparseState object.

        Returns:
            None
        """
        if isinstance(path_or_state, str):
            self._state = SparseState.load_from_file(path_or_state)
        else:
            self._state = path_or_state


    def save_state(self, path:str):
        """
        Save the state of the Tsetlin Machine to the specified file path.

        Parameters:
            path (str): The file path to save the state to.

        Returns:
            None
        """        
        if self._state is None:

            if self._cbs is not None:
                self._load_state_from_backend()
            else:
                raise ValueError("Cannot save a empty Tsetlin Machine without a stored state. Is the Tsetlin Machine trained?")
                    
        self._state.save_to_file(path)

    def get_predictor(self, explanation: str = "none", exclude_negative_clauses=False) -> "gt.Predictor":
        rs = gt.RuleSet(is_multi_label=self._is_multi_label)
        rs.compile_from_sparse_state(self._state)
        return gt.Predictor.from_ruleset(rs, explanation, exclude_negative_clauses)   

    def _set_extra_params_on_cb(self, cb, k:int):

        if self.active_literals_size < 1:
            raise ValueError("Active literals size must be greater than 0, current value: {}".format(self.active_literals_size))
        
        if self.clause_size < 1:
            raise ValueError("Clause size must be greater than 0, current value: {}".format(self.clause_size))

        # want this?
        if self.lower_ta_threshold > 0:
            raise ValueError("Cannot have a positive lower_ta_threshold ({}). Should be set to a negative number.".format(self.lower_ta_threshold))


        cb.set_active_literals_size(self.active_literals_size)
        cb.set_clause_size(self.clause_size)
        cb.set_lower_ta_threshold(self.lower_ta_threshold)


    def _get_backend(self):
        """
        Select the correct backend implementation based on the current settings.
        """

        imp_dict = {
            (True, True, True):     "sparse_cb_Lt_Dt_Bt",
            (True, True, False):    "sparse_cb_Lt_Dt_Bf",
            (True, False, True):    "sparse_cb_Lt_Df_Bt",
            (True, False, False):   "sparse_cb_Lt_Df_Bf",
            (False, True, True):    "sparse_cb_Lf_Dt_Bt",
            (False, True, False):   "sparse_cb_Lf_Dt_Bf",
            (False, False, True):   "sparse_cb_Lf_Df_Bt",
            (False, False, False):  "sparse_cb_Lf_Df_Bf"
        }

        lb_temp = True
        if self.literal_budgets[0] is None:
            lb_temp = False

        # print(imp_dict[(lb_temp, self.dynamic_AL, self.boost_true_positives)])
        backend_cb = _backend_impl[imp_dict[(lb_temp, self.dynamic_AL, self.boost_true_positives)]]

        _backend_impl["sparse_cb"] = backend_cb

        return backend_cb


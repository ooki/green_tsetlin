
from typing import Union, Optional, List
import itertools
from contextlib import contextmanager
import copy

import numpy as np

import green_tsetlin as gt
from green_tsetlin.backend import impl as _backend_impl
import pickle

class DenseState:
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


    def copy(self) -> "DenseState":
        tm = DenseState()
        tm.w = self.w.copy()
        tm.c = self.c.copy()
        return tm

    @staticmethod
    def load_from_file(file_path) -> "DenseState":
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        d = np.load(file_path)
        ds = DenseState()

        if d["w"].shape[0] != d["c"].shape[0]:
            raise ValueError("Cannot load state. w and c must have the same number of clauses")        

        if d["w"].dtype != np.int16:
            raise ValueError("Clause Weights much be np.int16 is {}".format(ds["w"].dtype))
        
        if d["c"].dtype != np.int8:
            raise ValueError("Clause State much be np.int8 is {}".format(ds["c"].dtype))

        ds.w = d["w"]
        ds.c = d["c"]
        return ds

    def save_to_file(self, file_path) -> None:
        if not file_path.endswith(".npz"):
            raise ValueError("State object must be a .npz file")
        
        np.savez(file_path, w=self.w, c=self.c)



class TsetlinMachine:
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, s : Union[float, list], threshold: int,
                 literal_budget:Optional[int]=None, boost_true_positives: bool = False, multi_label:bool=False):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        
        self.n_classes = n_classes
        self._cbs : list = None
        self._state : DenseState = None                
                            
        self.boost_true_positives = boost_true_positives
        # if literal_budget is None:
        #     literal_budget = 32700
            
        # elif isinstance(literal_budget, list):
        #     raise ValueError("Cannot set a list-version of literal_budgets in the constructor. Use set_literal_budget() instead.")
                
        self.literal_budgets = [literal_budget] # high value, should really be set.
            

        self._clause_block_sizes:list = None
        self._trainable_flags:list = None

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

        # sparse specific
        # self._sparse_params_set = False
        # self.clause_size = n_literals
        # self.active_literals_size = n_literals
        # self.lower_ta_threshold = -20
        # self.dynamic_AL = False
        
        self._backend_clause_block_cls = _backend_impl["cb"]

    def set_trainable_flags(self, trainable_flags:List[bool]):
        """
        Set the trainable flags for the blocks of the clause.
        
        Args:
            trainable_flags (List[bool]): A list of boolean values indicating whether each clause block is trainable or not.

        Raises:
            ValueError: If the clause block sizes have not been set before setting the trainable flags, or if the number of trainable flags does not match the number of clause blocks.
        """

        if self._clause_block_sizes is None:
            raise ValueError("Cannot set the trainable flags before setting the clause blocks sizes (or number of blocks) first")

        if len(trainable_flags) != len(self._clause_block_sizes):
            raise ValueError("The number of trainable flags must match the number of clause blocks")

        self._trainable_flags = trainable_flags
        raise NotImplementedError("Not impl trainable flags in the backend yet!")
    

    def set_literal_budget(self, literal_budget_or_budgets_list: Union[int, list]):
        """Sets the number of literals to be used in the backend for this Tsetlin Machine.

        Args:
            literal_budget_or_budgets_list (Union[int, list]): The max number of ON literals in a clause.
                                                      If a list then len(literal_budget_or_budgets_list) number of budgets will be used, each with the size as specified in literal_budget_or_budgets_list.
                                                      So if literal_budget_or_budgets_list = [10, 20, 30] it will create 3 budgets with sizes 10, 20 and 30.
                                                      If the len of literal_budget_or_budgets_list is not equal to the number of blocks then the budget list will repeat (same as with s).
                                                      So if literal_budget_or_budgets_list = [10, 20, 30] and we have 5 clause blocks then the budgets used will be [10, 20, 30, 10, 20].

        """
        if isinstance(literal_budget_or_budgets_list, int):
            self.literal_budgets = [literal_budget_or_budgets_list]
        else:
            self.n_literals_budget = literal_budget_or_budgets_list

        for budget in self.literal_budgets:
            if budget < 1:
                raise ValueError("Cannot have a non-positive budget ({})".format(budget))                                                        


    def set_num_clause_blocks(self, n_blocks_or_cb_sizes: Union[int, list]):
        """Sets the number of blocks to be used in the backend for this Tsetlin Machine.

        Args:
            n_blocks_or_cb_sizes (Union[int, list]): n_blocks_or_cb_sizes (int or list): The number of blocks to create. If the reminder is not zero it will be added to the first block.
                                                      If a list then len(n_blocks_or_cb_sizes) number of blocks will be created, each with the size as specified in clause_block_sizes.
                                                      So if n_blocks_or_cb_sizes = [10, 20, 30] it will create 3 blocks with sizes 10, 20 and 30.
                                                      The total number of clauses will have to match the number of clauses in the Tsetlin Machine.

        """
        if isinstance(n_blocks_or_cb_sizes, int):
            n_blocks = n_blocks_or_cb_sizes

            n_clause_per_block = self.n_clauses // n_blocks
            n_add = self.n_clauses % n_blocks

            self._clause_block_sizes = [n_clause_per_block for _ in range(n_blocks)]
            self._clause_block_sizes[0] += n_add

        else:            
            if sum(n_blocks_or_cb_sizes) != self.n_clauses:
                raise ValueError("The sum of the clause block sizes ({}) does not match the number of clauses in the Tsetlin Machine {}".format(sum(n_blocks_or_cb_sizes), self.n_clauses))
            
            self._clause_block_sizes = n_blocks_or_cb_sizes
        
        if any(cb_size < 1 for cb_size in self._clause_block_sizes):
            raise ValueError("Cannot have a clause block size under 1 (currently: {}))".format(self._clause_block_sizes))
        

    def _load_state_from_backend(self, only_return_copy:bool=False):
        """ Collects the state from the backend.
        if only_return_copy is True => just return the state, else set it to state_ in the TM                        
        """
        
        state = self._state
        if only_return_copy:
            state = None
        
        if state is None: # allocate state
            state = DenseState(n_literals=self.n_literals, n_clauses=self.n_clauses, n_classes=self.n_classes)
        
        clause_offset = 0 
        for cb in self._cbs:
            cb.get_clause_state(state.c, clause_offset)
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
        for cb in self._cbs:
            cb.set_clause_state(self._state.c, clause_offset)
            cb.set_clause_weights(self._state.w, clause_offset)
            clause_offset += cb.get_number_of_clauses()


    def load_state(self, path_or_state: Union[str, DenseState]):
        """
        Load the state from the given path or state object.

        Parameters:
            path_or_state (Union[str, DenseState]): The path to the state file or the DenseState object.

        Returns:
            None
        """
        if isinstance(path_or_state, str):
            self._state = DenseState.load_from_file(path_or_state)
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

    def get_ruleset(self):        
        rs = gt.RuleSet(self._is_multi_label)
        rs.compile_from_dense_state(self._state)        
        return rs
    
    def construct_clause_blocks(self) -> list:
        """ Creates the backend clause blocks, will not allocate them (use allocate_clause_blocks Context Manager).
        The clause blocks are found in the _cbs attribute, and a copy of this list returned (sharing the cb's, but not the list).

        Args:
            
        Returns:
            list: A list of clause_blocks of type self._backend_clause_block_cls, the list is a copy while the members are shared.
        """

        trainable_flags = self._trainable_flags
        if trainable_flags is None:
            trainable_flags = [True] * len(self._clause_block_sizes)
            

        self._backend_clause_block_cls = self._get_backend()

        self._cbs = []
        for k, (s_k, literal_budget, n_clauses_in_block, is_trainable) in enumerate(zip(itertools.cycle(self.s), itertools.cycle(self.literal_budgets), self._clause_block_sizes, trainable_flags)):            
            cb = self._backend_clause_block_cls(self.n_literals, n_clauses_in_block, self.n_classes)
            
            cb.set_s(s_k)
            if self.literal_budgets[0] is not None:
                cb.set_literal_budget(literal_budget)
            

            # cb.set_trainable(is_trainable) # TODO: add in backend
            self._set_extra_params_on_cb(cb, k)
            self._cbs.append(cb)
        
        return copy.copy(self._cbs)
    

    def get_predictor(self, explanation: str = "none", exclude_negative_clauses=False) -> "gt.Predictor":
        rs = gt.ruleset.RuleSet(is_multi_label=self._is_multi_label)
        rs.compile_from_dense_state(self._state)        
        return gt.Predictor.from_ruleset(rs, explanation, exclude_negative_clauses)
    


    def _set_extra_params_on_cb(self, cb, k:int):
        pass


    def _get_backend(self):
       
        imp_dict = {
            (True, True): "cb_Lt_Bt",
            (True, False): "cb_Lt_Bf",
            (False, True): "cb_Lf_Bt",
            (False, False): "cb_Lf_Bf"
        }


        lb_temp = True
        if self.literal_budgets[0] is None:
            lb_temp = False


        # print(imp_dict[(lb_temp, self.boost_true_positives)])
        backend_cb = _backend_impl[imp_dict[(lb_temp, self.boost_true_positives)]]
       
        _backend_impl["cb"] = backend_cb
        return backend_cb








class ConvolutionalTsetlinMachine(TsetlinMachine):
    def __init__(self, n_literals:int, n_clauses: int, n_classes: int, s : Union[float, list], threshold: int,
                 patch_width:int, patch_height:int,
                 literal_budget:Optional[int]=None, multi_label:bool=False):
        
        super().__init__(n_literals, n_clauses, n_classes, s, threshold, literal_budget, multi_label)
        
        self.patch_width = patch_width
        self.patch_height = patch_height
        
    def _get_backend(self):
        """
        Select the correct backend implementation based on the current settings.
        """
        
        return _backend_impl["conv_cb"]
    
        

        

@contextmanager
def allocate_clause_blocks(cbs_or_tm: Union[list, TsetlinMachine] , seed: int):
    """ Allocate the clause blocks on entry, will deallocate the blocks on exit.

    Args:
        cbs_or_tm (Union[list, TsetlinMachine]): Either a list of clause blocks or a TsetlinMachine.
        seed (int): Seed for the clause blocks, each clause block will get a unique seed starting from seed to seed+len(cbs).

    Raises:
        ValueError: If a TsetlinMachine has no clause blocks or the list is empty.
    """
    if isinstance(cbs_or_tm, TsetlinMachine):
        cbs = cbs_or_tm._cbs
        if cbs is None:
            raise ValueError("The Tsetlin Machine has no clause blocks. Did you call constrcut_clause_blocks() first?")        
    else:
        cbs = cbs_or_tm
        if len(cbs) == 0:
            raise ValueError("The provided clause block list is empty, cannot initialize a empty list.")


    running_seed = seed
    for cb in cbs:
        cb.initialize(seed=running_seed)
        running_seed += 1

    yield

    for cb in cbs:
        cb.cleanup()

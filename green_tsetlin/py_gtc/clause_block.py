
import numpy 

from green_tsetlin.py_gtc.feedback_block import FeedbackBlock
from green_tsetlin.py_gtc.dense_input_block import DenseInputBlock, SparseInputBlock, SparseInpuDenseOutputBlock
from green_tsetlin.py_gtc.tsetlin_state import TsetlinState, TsetlinStateSparse
import numpy as np 


class ClauseBlock:
    def __init__(self, n_literals, n_clauses, n_classes):
        self.m_is_init = False
        self.m_is_trainable = True # unsure
        self.m_input_block = None
        self.m_feedback_block = None

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        # this is not clear
        self.state = TsetlinState(n_literals,
                                    n_clauses,
                                    n_classes)
    
    
    def initialize(self, seed):
        self.m_is_init = True
        self.state.initialize(seed=seed)

    def is_init(self):
        return self.m_is_init
    
    def get_input_block(self):
        return self.m_input_block

    def get_feedback(self):
        return self.m_feedback_block
    
    def set_trainable(self, is_trainable):
        self.m_is_trainable = is_trainable
    
    def is_trainable(self):
        return self.m_is_trainable

    def cleanup(self):
        
        self.state.cleanup()
        self.m_is_init = False
    
    def set_s(self, s):
        self.state.s = s
    
    def set_literal_budget(self, budget):
        self.state.literal_budget = budget

    
    def set_feedback(self, feedback_block):

        if isinstance(feedback_block, FeedbackBlock):
            self.m_feedback_block = feedback_block
        else:
            raise ValueError("FeedbackBlock object expected.")


    def get_feedback(self):
        return self.m_feedback_block


    def set_input_block(self, ib):

        if isinstance(ib, DenseInputBlock) or isinstance(ib, SparseInpuDenseOutputBlock):
            self.m_input_block = ib
        else:
            raise ValueError("DenseInputBlock object or SparseInpuDenseOutputBlock expected, got: {}".format(type(ib)))

    def train_example(self):

        self.pull_example()
        self.train_set_clause_output()
        self.set_votes()

    def eval_example(self):
        self.pull_example()
        self.eval_set_clause_output()
        self.set_votes()

    def train_update(self, positive_class, prob_positive, negative_class, prob_negative):
        self.state.train_update(self.m_literals, positive_class, prob_positive, negative_class, prob_negative, self.clause_outputs)

    def pull_example(self):
        self.m_literals = self.m_input_block.pull_current_example()
    
    def train_set_clause_output(self):
        self.clause_outputs = self.state.set_clause_output(self.m_literals, 0)

    def eval_set_clause_output(self):
        self.clause_outputs = self.state.eval_clause_output(self.m_literals, 0)
        
    def set_votes(self):
        self.clause_votes = self.state.vote_counter(self.clause_outputs)
        self.m_feedback_block.register_votes(self.clause_votes)

    def get_number_of_clauses(self):
        return self.n_classes

    def get_clause_state(self, c_copy, clause_offset):
        np.copyto(c_copy, self.state.get_clause_state(c_copy, clause_offset))

    def get_clause_weights(self, w_copy, clause_offset):
        np.copyto(w_copy, self.state.get_clause_weights(w_copy, clause_offset))

    def get_number_of_clauses(self):
        return self.state.n_clauses
        

class ClauseBlockSparse(ClauseBlock):
    def __init__(self, n_literals, n_clauses, n_classes):
        super().__init__(n_literals, n_clauses, n_classes)
        self.state = TsetlinStateSparse(n_literals,
                                          n_clauses,
                                          n_classes)
        

    def initialize(self, seed):
        self.m_is_init = True
        self.state.initialize(seed=seed)


    def set_input_block(self, ib):

        if isinstance(ib, SparseInputBlock):
            self.m_input_block = ib
        else:
            raise ValueError("SparseInputBlock object expected.")
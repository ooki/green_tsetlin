import numpy 
from green_tsetlin.py_gtc.feedback_block import FeedbackBlock
from green_tsetlin.py_gtc.dense_input_block import DenseInputBlock
from green_tsetlin.py_gtc.py_tsetlin_machine import pyTsetlin
import numpy as np 

class ClauseBlock:
    def __init__(self, n_literals, n_clauses, n_classes):
        self.m_is_init = False
        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        # this is not clear
        self.state = pyTsetlin(n_literals,
                               n_clauses,
                               n_classes)
    
    
    def initialize(self, seed):
        self.m_is_init = True
        self.state.initialize(seed=0)


    def is_init(self):
        return self.m_is_init
    
    def cleanup(self):
        pass
    
    def set_s(self, s):
        pass
    
    def set_literal_budget(self, budget):
        pass

    

    def set_feedback(self, feedback_block):

        if isinstance(feedback_block, FeedbackBlock):
            self.m_feedback_block = feedback_block
        else:
            raise ValueError("FeedbackBlock object expected.")


    def get_feedback(self):
        return self.m_feedback_block


    def set_input_block(self, ib):

        if isinstance(ib, DenseInputBlock):
            self.m_input_block = ib
        else:
            raise ValueError("DenseInputBlock object expected.")

    def train_example(self):

        self.pull_example()
        self.train_set_clause_output()
        self.set_votes()

    def eval_example(self):
        self.pull_example()
        self.eval_set_clause_output()
        self.set_votes()

    
    def pull_example(self):
        self.m_literals = self.m_input_block.pull_current_example()
    
    def train_set_clause_output(self):
        self.clause_outputs = self.state.set_clause_output(self.m_literals, 0)

    def eval_set_clause_output(self):
        self.clause_outputs = self.state.eval_clause_output(self.m_literals, 0)
        

    def set_votes(self):
        self.clause_votes = self.state.vote_counter(self.clause_outputs)
        self.m_feedback_block.register_votes(self.clause_outputs)

    def get_clause_state(self, c_copy, clause_offset):
        np.copyto(c_copy, self.state.get_clause_state())

    def get_clause_weights(self, w_copy, clause_offset):
        np.copyto(w_copy, self.state.get_clause_weights())

    def get_number_of_clauses(self):
        return self.state.n_clauses

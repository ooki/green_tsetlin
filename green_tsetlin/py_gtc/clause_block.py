import numpy 
from green_tsetlin.py_gtc.feedback_block import FeedbackBlock
from green_tsetlin.py_gtc.dense_input_block import DenseInputBlock


class ClauseBlock:
    def __init__(self, n_literals, n_clauses, n_classes):
        pass
    
    def initialize(self, seed):
        pass
    
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
        # train_set_clause_output()
        # set_votes()

    def pull_example(self):
        self.m_literals = self.m_input_block.pull_current_example()
        
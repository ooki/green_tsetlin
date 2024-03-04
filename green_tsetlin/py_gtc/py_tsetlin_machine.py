
import numpy as np


class pyTsetlin:
    def __init__(self, n_literals, n_clauses, n_classes):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        self.clauses = None
        self.clause_weights = None


    def initialize(self, seed):
        
        self.clauses = np.random.choice(np.array([-1, 0]), size=(self.n_clauses, self.n_literals*2), replace=True).astype(np.int8)
        self.clause_weights = np.random.choice(np.array([-1, 1]), size=(self.n_classes, self.n_clauses), replace=True).astype(np.int8)



    def set_clause_output(self, m_literals, seed):

        # temp
        return np.zeros(self.n_clauses, dtype=np.uint8)
    

    def vote_counter(self, clause_outputs):

        # TODO Multi lable structure 
        return np.dot(self.clause_weights, clause_outputs)

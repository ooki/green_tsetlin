
import numpy as np


class pyTsetlin:
    def __init__(self, n_literals, n_clauses, n_classes):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        self.clauses = None
        self.clause_weights = None

        self.literal_counts = np.zeros(self.n_clauses, dtype=np.int32)

    def initialize(self, seed):
        
        self.clauses = np.random.choice(np.array([-1, 0]), size=(self.n_clauses, self.n_literals*2), replace=True).astype(np.int8)
        self.clause_weights = np.random.choice(np.array([-1, 1]), size=(self.n_clauses, self.n_classes), replace=True).astype(np.int8)



    def set_clause_output(self, m_literals, seed):

        clause_outputs = np.zeros(self.n_clauses, dtype=np.uint8)
        
        for clause_k in range(self.n_clauses):

            pos_literal_count = 0
            neg_literal_count = 0
            
            clause_outputs[clause_k] = 1


            for idx in range(self.n_literals):
                # pos side
                if(self.clauses[clause_k][idx] > 0):
                    if(m_literals[idx] == 0):
                        clause_outputs[clause_k] = 0
                        break

                    if(0): # do_literal_budget
                        pos_literal_count += 1

                # neg side
                if(self.clauses[clause_k][idx + self.n_literals] > 0):
                    if(m_literals[idx] == 1): 
                        clause_outputs[clause_k] = 0
                        break

                    if(0): # do_literal_budget
                        neg_literal_count += 1


            if(0): # do_literal_budget
                self.literal_counts[clause_k] = pos_literal_count + neg_literal_count
        # return if clause evaluates to possitie or negative based on literals
        return clause_outputs
    

    def eval_clause_output(self, m_literals, seed):

        # temp
        return np.zeros(self.n_clauses, dtype=np.uint8)


    def vote_counter(self, clause_outputs):

        # TODO Multi lable structure 
        return np.dot(clause_outputs, self.clause_weights)

    def get_clause_state(self):
        return self.clauses
    
    def get_clause_weights(self):
        return self.clause_weights
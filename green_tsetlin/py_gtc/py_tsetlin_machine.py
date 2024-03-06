
import numpy as np


class pyTsetlinState:
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


            for literal_k in range(self.n_literals):

                # pos side
                if(self.clauses[clause_k][literal_k] >= 0):
                    if(m_literals[literal_k] == 0):
                        clause_outputs[clause_k] = 0
                        break

                    if(0): # do_literal_budget
                        pos_literal_count += 1

                # neg side
                if(self.clauses[clause_k][literal_k + self.n_literals] >= 0):
                    if(m_literals[literal_k] == 1): 
                        clause_outputs[clause_k] = 0
                        break

                    if(0): # do_literal_budget
                        neg_literal_count += 1


            if(0): # do_literal_budget
                self.literal_counts[clause_k] = pos_literal_count + neg_literal_count

        return clause_outputs
    
    def eval_clause_output(self, m_literals, seed):
        
        clause_outputs = np.zeros(self.n_clauses, dtype=np.uint8)

        for clause_k in range(self.n_clauses):
            
            clause_outputs[clause_k] = 1
            is_empty_clause = True

            for literal_k in range(self.n_literals):
                # pos side
                if(self.clauses[clause_k][literal_k] > 0):
                    is_empty_clause = False
                    if(m_literals[literal_k] == 0):
                        clause_outputs[clause_k] = 0
                        break
                
                # neg side
                if(self.clauses[clause_k][literal_k + self.n_literals] > 0):
                    is_empty_clause = False
                    if(m_literals[literal_k] == 1):    
                        clause_outputs[clause_k] = 0
                        break

            if(is_empty_clause):
                clause_outputs[clause_k] = 0

        return clause_outputs

    def train_update(self, literals, positive_class, prob_positive, negative_class, prob_negative, clause_outputs):

        for clause_k in range(self.n_clauses):

            if(0): # do_literal_budget
                pass

            if(np.random.random() < prob_positive):                           # * n_classes] + positive_class?
                self.update_clause(self.clauses[clause_k], self.clause_weights[clause_k], 1, literals, clause_outputs[clause_k], positive_class)

            if(np.random.random() < prob_negative):                           # * n_classes] + negative_class?
                self.update_clause(self.clauses[clause_k], self.clause_weights[clause_k], -1, literals, clause_outputs[clause_k], negative_class)

    
    def update_clause(self, clause_row, clause_weight, target, literals, clause_output, class_k):

        # here, what if clause_weight is 2d, does that even happen?
        sign = 1 if (clause_weight[class_k] >= 0) else -1

        if(target*sign > 0):
            if(clause_output == 1):
                clause_weight[class_k] += sign
                self.T1aFeedback(clause_row, literals)
                
            else:
                self.T1bFeedback(clause_row)

        elif((target*sign < 0) and clause_output == 1):
            clause_weight[class_k] -= sign
            self.T2Feedback(clause_row, literals)



    def T1aFeedback(self, clause_row, literals):

        use_boost_true_positive = False
        s_inv = (1.0 / self.s)
        s_min1_inv = (self.s - 1.0) / self.s

        lower_state = -127
        upper_state =  127


        for literal_k in range(self.n_literals):
            
            if(literals[literal_k] == 1):
                if(use_boost_true_positive):
                    if(clause_row[literal_k] < upper_state):
                        clause_row[literal_k] += 1
                # pos 1        
                else:
                    if(np.random.random() <= s_min1_inv):
                        if(clause_row[literal_k] < upper_state):
                            clause_row[literal_k] += 1
                
                # neg 1
                if(np.random.random() <= s_inv):
                    if(clause_row[literal_k + self.n_literals] > lower_state):
                        clause_row[literal_k + self.n_literals] -= 1
            
            else:
                # neg 2
                if(np.random.random() <= s_inv):
                    if(clause_row[literal_k] > lower_state):
                        clause_row[literal_k] -= 1

                # pos 2
                if(np.random.random() <= s_min1_inv):
                    if(clause_row[literal_k + self.n_literals] < upper_state):
                        clause_row[literal_k + self.n_literals] += 1


    def T1bFeedback(self, clause_row):

        s_inv = (1.0 / self.s)
        lower_state = -127

        for literal_k in range(self.n_literals):

            # pos 1
            if(np.random.random() <= s_inv):
                if(clause_row[literal_k] > lower_state):
                    clause_row[literal_k] -= 1
            
            # neg 1 
            if(np.random.random() <= s_inv):
                if(clause_row[literal_k + self.n_literals] > lower_state):
                    clause_row[literal_k + self.n_literals] -= 1



    def T2Feedback(self, clause_row, literals):

        for literal_k in range(self.n_literals):

            if(literals[literal_k] == 0):
                if(clause_row[literal_k] < 0):
                    clause_row[literal_k] += 1

            else:
                if(clause_row[literal_k + self.n_literals] < 0):
                    clause_row[literal_k + self.n_literals] += 1
  


    def vote_counter(self, clause_outputs):

        # TODO Multi lable structure 
        return np.dot(clause_outputs, self.clause_weights)

    def get_clause_state(self):
        return self.clauses
    
    def get_clause_weights(self):
        return self.clause_weights

import numpy as np


class TsetlinState:
    def __init__(self, n_literals, n_clauses, n_classes):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        self.clauses = None
        self.clause_weights = None
        self.literal_budget = None
        self.do_literal_budget = True

    def initialize(self, seed):
        
        self.rng = np.random.default_rng(seed)

        self.clauses = self.rng.choice(np.array([-1, 0]), size=(self.n_clauses, self.n_literals*2), replace=True).astype(np.int8)
        self.clause_weights = self.rng.choice(np.array([-1, 1]), size=(self.n_clauses, self.n_classes), replace=True).astype(np.int8)
        
        self.class_votes = np.zeros(self.n_classes, dtype=np.int32)
        self.literal_counts = np.zeros(self.n_clauses, dtype=np.int32)
        
        if(self.literal_budget is None or self.literal_budget == 32700):
            self.do_literal_budget = False

    def cleanup(self):
        self.clauses = None
        self.clause_weights = None
        self.class_votes = np.zeros(self.n_classes, dtype=np.int32)

        if(self.do_literal_budget):
            self.literal_counts = np.zeros(self.n_clauses, dtype=np.int32)



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

                    if(self.do_literal_budget): # do_literal_budget
                        pos_literal_count += 1

                # neg side
                if(self.clauses[clause_k][literal_k + self.n_literals] >= 0):
                    if(m_literals[literal_k] == 1): 
                        clause_outputs[clause_k] = 0
                        break

                    if(self.do_literal_budget): # do_literal_budget
                        neg_literal_count += 1


            if(self.do_literal_budget): # do_literal_budget
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

            if(self.do_literal_budget): # do_literal_budget
                if(self.literal_counts[clause_k] > self.literal_budget):
                    clause_outputs[clause_k] = 0
                    

            if(self.rng.random() < prob_positive):                           # * n_classes] + positive_class?
                self.update_clause(self.clauses[clause_k], self.clause_weights[clause_k], 1, literals, clause_outputs[clause_k], positive_class)

            if(self.rng.random() < prob_negative):                           # * n_classes] + negative_class?
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
                    if(self.rng.random() <= s_min1_inv):
                        if(clause_row[literal_k] < upper_state):
                            clause_row[literal_k] += 1
                
                # neg 1
                if(self.rng.random() <= s_inv):
                    if(clause_row[literal_k + self.n_literals] > lower_state):
                        clause_row[literal_k + self.n_literals] -= 1
            
            else:
                # neg 2
                if(self.rng.random() <= s_inv):
                    if(clause_row[literal_k] > lower_state):
                        clause_row[literal_k] -= 1

                # pos 2
                if(self.rng.random() <= s_min1_inv):
                    if(clause_row[literal_k + self.n_literals] < upper_state):
                        clause_row[literal_k + self.n_literals] += 1


    def T1bFeedback(self, clause_row):

        s_inv = (1.0 / self.s)
        lower_state = -127

        for literal_k in range(self.n_literals):

            # pos 1
            if(self.rng.random() <= s_inv):
                if(clause_row[literal_k] > lower_state):
                    clause_row[literal_k] -= 1
            
            # neg 1 
            if(self.rng.random() <= s_inv):
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

    def get_clause_state(self, src, clause_offset):
        src[clause_offset:self.n_clauses+clause_offset] = self.clauses
        return src
    
    def get_clause_weights(self, src, clause_offset):
        src[clause_offset:self.n_clauses+clause_offset] = self.clause_weights
        return src

class TsetlinStateSparse:
    def __init__(self, n_literals, n_clauses, n_classes):

        self.n_literals = n_literals
        self.n_clauses = n_clauses
        self.n_classes = n_classes

        self.clauses = None
        self.clause_indexes = None     

        self.clause_weights = None
        self.active_literals = None
        self.literal_budget = None
        self.do_literal_budget = True

        self.t = 20
        self.al_size = 7

    def initialize(self, seed):
        
        self.rng = np.random.default_rng(seed)

        self.clauses = [[] for _ in range(self.n_clauses*2)]      
        self.clause_indexes = [[] for _ in range(self.n_clauses*2)]        

        self.clause_weights = self.rng.choice(np.array([-1, 1]), size=(self.n_clauses, self.n_classes), replace=True).astype(np.int8)
        self.active_literals = [[] for _ in range(self.n_classes*2)]    

        self.class_votes = np.zeros(self.n_classes, dtype=np.int32)
        self.literal_counts = np.zeros(self.n_clauses, dtype=np.int32)
        
        if(self.literal_budget is None or self.literal_budget == 32700):
            self.do_literal_budget = False

    def cleanup(self):
        self.clauses = None
        self.clause_indexes = None     

        self.active_literals = None
        self.clause_weights = None
        self.class_votes = np.zeros(self.n_classes, dtype=np.int32)

        if(self.do_literal_budget):
            self.literal_counts = np.zeros(self.n_clauses, dtype=np.int32)

    def sort_lists(self, clause_k):

        if len(self.clause_indexes[clause_k]) == 0:
            return
    

        zipped_lists = list(zip(self.clause_indexes[clause_k], self.clauses[clause_k]))
        zipped_lists.sort()
        sorted_indexes, sorted_states = zip(*zipped_lists)

        
        self.clause_indexes[clause_k] = list(sorted_indexes)
        self.clauses[clause_k] = list(sorted_states)
        

    def set_clause_output(self, m_literals, seed):
        
        clause_outputs = np.ones(self.n_clauses, dtype=np.int32)


        for clause_k in range(self.n_clauses):

            if (len(self.clauses[clause_k]) == 0) and (len(self.clauses[clause_k + self.n_classes]) == 0):
                continue
                
            else:  
                for i, ta in enumerate(self.clause_indexes[clause_k]):
                    if self.clauses[clause_k][i] > 0 and (ta not in m_literals):
                        clause_outputs[clause_k] = 0

                for i, ta in enumerate(self.clause_indexes[clause_k + self.n_clauses]):
                    if self.clauses[clause_k + self.n_clauses][i] > 0 and (ta in m_literals):
                        clause_outputs[clause_k] = 0


        return clause_outputs
    


    def eval_clause_output(self, m_literals, seed):

        clause_outputs = np.ones(self.n_clauses, dtype=np.int32)

        for clause_k in range(self.n_classes):
            
            for i, ta in enumerate(self.clause_indexes[clause_k]):
                    if self.clauses[clause_k][i] > 0 and (ta not in m_literals):
                        clause_outputs[clause_k] = 0

            for i, ta in enumerate(self.clause_indexes[clause_k + self.n_clauses]):
                if self.clauses[clause_k + self.n_clauses][i] > 0 and (ta in m_literals):
                    clause_outputs[clause_k] = 0

        return clause_outputs
    

    def vote_counter(self, clause_outputs):

        # TODO Multi lable structure 
        return np.dot(clause_outputs, self.clause_weights)
    
    def train_update(self, literals, positive_class, prob_positive, negative_class, prob_negative, clause_outputs):

        for clause_k in range(self.n_clauses):
            
            if(self.do_literal_budget): # do_literal_budget
                if(self.literal_counts[clause_k] > self.literal_budget):
                    clause_outputs[clause_k] = 0
                    

            if(self.rng.random() < prob_positive):                           # * n_classes] + positive_class?
                self.update_clause(clause_k, self.clause_weights[clause_k], 1, literals, clause_outputs[clause_k], positive_class)

            if(self.rng.random() < prob_negative):                           # * n_classes] + negative_class?
                self.update_clause(clause_k, self.clause_weights[clause_k], -1, literals, clause_outputs[clause_k], negative_class)

            i = 0  
            for state in self.clauses[clause_k][:]:
                if state < 0 - self.t:
                    self.clauses[clause_k].remove(state)
                    self.clause_indexes[clause_k].pop(i)
                else:
                    i += 1
            
            i = 0  
            for state in self.clauses[clause_k + self.n_clauses][:]:
                if state < 0 - self.t:
                    self.clauses[clause_k + self.n_clauses].remove(state)
                    self.clause_indexes[clause_k + self.n_clauses].pop(i)
                else:
                    i += 1
    

    def update_clause(self, clause_k, clause_weight, target, literals, clause_output, class_k):
        # here, what if clause_weight is 2d, does that even happen?
        sign = 1 if (clause_weight[class_k] >= 0) else -1

        if(target*sign > 0):
            if(clause_output == 1):
                clause_weight[class_k] += sign
                self.T1aFeedback(clause_k, literals, class_k)
                
            else:
                self.T1bFeedback(clause_k)



        elif((target*sign < 0) and clause_output == 1):
            clause_weight[class_k] -= sign
            self.T2Feedback(clause_k,  literals, class_k)


    def T1aFeedback(self, clause_k, literals, class_k):

        use_boost_true_positive = False
        s_inv = (1.0 / self.s)
        s_min1_inv = (self.s - 1.0) / self.s

        lower_state = -127
        upper_state =  127


        for lit in literals:
            if lit in self.clause_indexes[clause_k]:
                if(self.rng.random() <= s_min1_inv):
                    lit_index = int(np.where(self.clause_indexes[clause_k] == lit)[0])
                    self.clauses[clause_k][lit_index] += 1

            else:
                self.update_al(class_k, lit, True)


            if lit in self.clause_indexes[clause_k + self.n_clauses]:
                if(self.rng.random() <= s_inv): 
                    lit_index = int(np.where(self.clause_indexes[clause_k + self.n_clauses] == lit)[0])
                    self.clauses[clause_k + self.n_clauses][lit_index] -= 1

            else:
                self.update_al(class_k + self.n_classes, lit, True)

                
        for i, ta in enumerate(self.clause_indexes[clause_k]):
            
            if ta not in literals:
                if(self.rng.random() <= s_inv): 
                    self.clauses[clause_k][i] -= 1

        for i, ta in enumerate(self.clause_indexes[clause_k + self.n_clauses]):

            if ta not in literals:
                if(self.rng.random() <= s_min1_inv):
                    self.clauses[clause_k + self.n_clauses][i] += 1


    def T1bFeedback(self, clause_k):

        s_inv = (1.0 / self.s)
        lower_state = -127
        
        for idx in range(len(self.clause_indexes[clause_k])):
            if(self.rng.random() <= s_inv): 
                self.clauses[clause_k][idx] -= 1

        for idx in range(len(self.clause_indexes[clause_k + self.n_clauses])):
            if(self.rng.random() <= s_inv): 
                self.clauses[clause_k + self.n_clauses][idx] -= 1


    def T2Feedback(self, clause_k, literals, class_k):

        for idx, ta in enumerate(self.clause_indexes[clause_k]):
            if ta not in literals:
                if self.clauses[clause_k][idx] <= 0:
                    self.clauses[clause_k][idx] += 1
            
        
        for idx, ta in enumerate(self.clause_indexes[clause_k + self.n_clauses]):
            if ta in literals:
                if self.clauses[clause_k + self.n_clauses][idx] <= 0:
                    self.clauses[clause_k + self.n_clauses][idx] += 1



        for lit in self.active_literals[class_k]:
            if(lit not in self.clause_indexes[clause_k]) and (lit not in literals):
                self.clauses[clause_k].append(0)
                self.clause_indexes[clause_k].append(lit)


        for lit in self.active_literals[class_k + self.n_classes]: 
            if(lit not in self.clause_indexes[clause_k + self.n_clauses]) and (lit in literals):
                self.clauses[clause_k + self.n_clauses].append(0)
                self.clause_indexes[clause_k + self.n_clauses].append(lit)



    def update_al(self, target, literal, dynamic_AL):
        if len(self.active_literals[target]) < self.al_size: # should be user defined
                if literal not in self.active_literals[target]:
                    self.active_literals[target].append(literal)
  
          
        else:
            if dynamic_AL:
                if literal not in self.active_literals[target]:
                    self.active_literals[target].pop(0)
                    self.active_literals[target].append(literal)
                

                    


    def get_clause_state(self, src, clause_offset):
        # TODO make the sparse rep -> dense
        # src[clause_offset:self.n_clauses+clause_offset] = self.clauses
        return src
    
    def get_clause_weights(self, src, clause_offset):
        src[clause_offset:self.n_clauses+clause_offset] = self.clause_weights
        return src
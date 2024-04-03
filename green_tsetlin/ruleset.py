
import numpy as np
import warnings



class RuleSet:
    def __init__(self, is_multi_label:bool):
        self.rules:list = None
        self.weights:np.array = None
        self.is_multi_label = is_multi_label

        self.n_literals = -1
        self.n_classes = -1



    def compile_from_dense_state(self, state):
        w = state.w
        c = state.c

        self.n_literals = c.shape[1] // 2
        self.n_classes = w.shape[1]

        rule_map = {}
        for k, row in enumerate(c):
            on_literals = np.nonzero(row >= 0)[0]
            if len(on_literals) < 1:
                continue
            
            key = tuple(on_literals)
            weights = rule_map.get(key, np.zeros(self.n_classes, dtype=np.int16))  
            weights += w[k]                                            
            
            if (w[k] == 0).all():
                continue
            
            rule_map[key] = weights

        self.rules = [list(k) for k in rule_map.keys()]
        self.weights = [w.tolist() for w in rule_map.values()]

    def compile_from_sparse_state(self, state):
        w = state.w
        _c_data = state.c_data
        _c_indices = state.c_indices
        _c_indptr = state.c_indptr

        if state.n_literals is not None:
            self.n_literals = state.n_literals
        else:
            self.n_literals = _c_indices[0].max() + 1
            warnings.warn('n_literals not set in state, using max index from c_indices as n_literals. Please verify this is correct, n_literals = {}.'.format(self.n_literals), stacklevel=2)
        self.n_classes = w.shape[1]

        # loop over all clauses 
        # get literals for given clause, get weights for given clause
        # add weights to the corresponding rule

        rule_map = {}
        done_clauses = 0
        for cb in range(len(_c_data)):
            
            c_data = _c_data[cb]
            c_indices = _c_indices[cb]
            c_indptr = _c_indptr[cb]

            num_clauses = (c_indptr.shape[0] - 1)//2

            for i in range(num_clauses):
                
                start = c_indptr[i]
                end = c_indptr[i+1]
                start_negated = c_indptr[i + num_clauses]
                end_negated = c_indptr[i + num_clauses + 1]
                
                clause_pos = c_indices[start:end]
                clause_pos_states = c_data[start:end]        
                clause_pos = [c for i, c in enumerate(clause_pos) if clause_pos_states[i] > 0]


                clause_neg = c_indices[start_negated:end_negated]
                clause_neg_states = c_data[start_negated:end_negated]
                clause_neg = [c+self.n_literals for i, c in enumerate(clause_neg) if clause_neg_states[i] > 0]
                
                weights = w[done_clauses]
                done_clauses += 1
                if (weights == 0).all():
                    continue
            
                clause_pos.extend(clause_neg)
                if len(clause_pos) < 1:
                    continue

                key = tuple(clause_pos)
                rule_weights = rule_map.get(key, np.zeros(self.n_classes, dtype=np.int16))
                rule_weights += weights

                rule_map[key] = rule_weights


        self.rules = [list(k) for k in rule_map.keys()]
        self.weights = [w.tolist() for w in rule_map.values()]

    def __len__(self):
        return len(self.rules)


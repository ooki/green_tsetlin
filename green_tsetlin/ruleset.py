
import numpy as np



class RuleSet:
    def __init__(self, is_multi_label:bool):
        self.rules:list = None
        self.weights:np.array = None
        self.is_multi_label = is_multi_label

        self.n_classes = -1


    def compile_from_dense_state(self, state):
        w = state.w
        c = state.c

        self.n_classes = w.shape[1]
        self.n_clauses = c.shape[0]

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
        self.weights = np.array([w.tolist() for w in rule_map.values()], dtype=np.int32)

    def __len__(self):
        return len(self.rules)




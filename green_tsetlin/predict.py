
from typing import Optional, Tuple, Union

import numpy as np

import green_tsetlin as gt
from green_tsetlin.backend import impl





class Predictor:
    def __init__(self, multi_label:bool=False, empty_class_output:Optional[int]=None):        
        self.is_multi_label = multi_label
        self.empty_class_output = empty_class_output
        
        self.n_literals: int = -1
        self.n_clauses: int = -1
        self.n_classes: int = -1
        self.n_features: int = -1
        
        # pickle support
        self.empty_class_output = empty_class_output
        self._raw_rules_cache = None
        self._raw_weights_cache = None
        self._raw_features_by_clause = None
        
    
    def _create_rules(self, state:gt.TMState):
        w = state.w
        c = state.c
        
        self.n_clauses, self.n_classes = w.shape[0], w.shape[1]
        self.n_literals = c.shape[1] // 2
        assert self.n_clauses == c.shape[0]               

        # if feature_map is not not enable Feature Importance
        # if feature_map is None:            
        #     feature_map = list(range(self.n_literals))
                            
        rules = {}
        for k, row in enumerate(c):
            on_literals = np.nonzero(row >= 0)[0]
            if len(on_literals) < 1:
                continue
            
            key = tuple(on_literals)
            weights = rules.get(key, np.zeros(self.n_classes, dtype=np.int16))  
            weights += w[k]                                            
            
            if (w[k] == 0).all():
                continue
            
            rules[key] = weights
            
        raw_rules = [list(k) for k in rules.keys()]
        raw_weights = [w.tolist() for w in rules.values()]

        # features_by_clause = []
        # if len(feature_map) != self.n_literals:
        #     raise ValueError("feature_map ({}) must be of length n_literals ({}).".format(len(feature_map, self.n_literals)))

        # for rule in raw_rules:
        #     features_in_rule = set(feature_map[lit_k] if lit_k < self.n_literals else (lit_k - self.n_literals) for lit_k in rule)
        #     features_by_clause.append(list(features_in_rule))
                
        # self.n_features = max(feature_map) + 1        
        self.n_clauses = len(raw_rules)
                

        self._raw_rules_cache = raw_rules
        self._raw_weights_cache = raw_weights
        # self._raw_features_by_clause = features_by_clause        
        

    
    @staticmethod   
    def create(tm:gt.TsetlinMachine) -> "Predictor":        
        predictor = Predictor(multi_label=tm._is_multi_label)
        predictor._create_rules(tm._state)
        
        
        
        
        
        
# support_literal_importance:bool=False, pickle_support:bool=True, empty_class_output:Optional[int]=None, multi_label:bool=False




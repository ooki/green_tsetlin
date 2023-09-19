
from typing import Optional, Tuple, Union

import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc


class RulePredictor:
    def __init__(self, support_literal_importance:bool=False, pickle_support:bool=True, empty_class_output:Optional[int]=None, multi_label:bool=False):
        self._inference : gtc.Inference = None
        self.support_literal_importance = support_literal_importance
        self.pickle_support = pickle_support
        self.multi_label = multi_label

        # used for pickle support
        self.empty_class_output = empty_class_output
        self._raw_rules_cache = None
        self._raw_weights_cache = None
        self._raw_features_by_clause = None

        self.n_literals: int = -1
        self.n_clauses: int = -1
        self.n_classes: int = -1
        self.n_features: int = -1


    def __getstate__(self):
        if self.multi_label:
            self.n_classes = self.n_classes * 2 # make sure we handle classes in the multi scenario correct

        state = self.__dict__.copy()
        del state["_inference"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # if self.multi_label is True:
        #     self.n_classes = 

        self._create_inference_object()
    
    def create_from_state(self, state, feature_map: Optional[list] = None):
        w = state["w"]
        c = state["c"]
        
        self.n_clauses, self.n_classes = w.shape[0], w.shape[1]
        self.n_literals = c.shape[1] // 2
        assert self.n_clauses == c.shape[0]               

        if feature_map is None:            
            feature_map = list(range(self.n_literals))
                            
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

        features_by_clause = []

        if len(feature_map) != self.n_literals:
            raise ValueError("feature_map ({}) must be of length n_literals ({}).".format(len(feature_map, self.n_literals)))

        for rule in raw_rules:
            features_in_rule = set(feature_map[lit_k] if lit_k < self.n_literals else (lit_k - self.n_literals) for lit_k in rule)
            features_by_clause.append(list(features_in_rule))

                
        self.n_features = max(feature_map) + 1        
        self.n_clauses = len(raw_rules)
                

        self._raw_rules_cache = raw_rules
        self._raw_weights_cache = raw_weights
        self._raw_features_by_clause = features_by_clause

        self._create_inference_object()

        if self.pickle_support is False:
            self._raw_rules_cache = None
            self._raw_weights_cache = None 
            self._raw_features_by_clause = None


    def _create_inference_object(self):
        """
        Creates and set the inference backend object.
        """
        if self.support_literal_importance:
            self._inference = gtc.Inference(self.n_literals, self.n_clauses, self.n_classes, self.n_features)             
        else:
            self._inference = gtc.InferenceNoLiteralsImportance(self.n_literals, self.n_clauses, self.n_classes, self.n_features)            
                           
        self._inference.set_rules_and_features(self._raw_rules_cache, self._raw_weights_cache, self._raw_features_by_clause)


        # shorten down - make sure we increase (2x) when we pickle for correct handling
        if self.multi_label is True:
            self.n_classes = self.n_classes // 2

                
        if self.empty_class_output is None:
            self.empty_class_output = self._inference.get_empty_class_output()                                        
        else:
            self._inference.set_empty_class_output(self.empty_class_output)


        


        
     

    def predict(self, x : np.ndarray, explain:bool=False, normalize_explaination:bool=True, literal_importance:bool=False) -> Union[int,  Tuple[int, list]]:
        
        if self.multi_label is False:
            y = self._inference.predict(x)
        else:
            y = self._inference.predict_multi(x)

        if explain is False:
            return y
        
        else:
            if literal_importance and self.support_literal_importance is False:
                raise ValueError("Cannot request literal importance on a Rules object that has support_literal_importance set to False.")        

            return y, np.array(self._inference.calc_local_importance(y, normalize_explaination))
        
    def explain(self,  x : np.ndarray, target_or_targets: Union[int, list], normalize:bool=True, literal_importance:bool=False):
        self._inference.predict(x)
        
        
        if isinstance(target_or_targets, int):            
            if literal_importance is False:
                return self._inference.calc_local_importance(target_or_targets, normalize)    
            else:
                if self.support_literal_importance is False:
                    raise ValueError("Cannot request literal importance on a Rules object that has support_literal_importance set to False.")
        
                f_imp = self._inference.calc_local_importance(target_or_targets, normalize)
                l_imp = self._inference.get_cached_literal_importance()

                return f_imp, l_imp
            
                        
        else:
            if literal_importance is False:
                return [np.array(self._inference.calc_local_importance(target, normalize)) for target in target_or_targets]
            
            else:
                if self.support_literal_importance is False:
                    raise ValueError("Cannot request literal importance on a Rules object that has support_literal_importance set to False.")
                
                expl = []
                for target in target_or_targets:
                    f_imp = self._inference.calc_local_importance(target, normalize)
                    l_imp = self._inference.get_cached_literal_importance()
                    expl.append((f_imp, l_imp))

                return expl
            
    def get_active_rules(self) -> np.array:
        return self._inference.get_active_clauses()
            
    def get_votes(self) -> np.array:
        return self._inference.get_votes()
        
    def get_global_importance(self, target_class, normalize:bool=True, literal_importance:bool=False):
        if literal_importance is False:
            return np.array(self._inference.calculate_global_importance(target_class, normalize))
        else:
            if self.support_literal_importance is False:
                raise ValueError("Cannot request literal importance on a Rules object that has support_literal_importance set to False.")
            
            f_imp = self._inference.calculate_global_importance(target_class, normalize)
            l_imp = self._inference.get_cached_literal_importance()
            return f_imp, l_imp


        
if __name__ == "__main__":
    n_literals = 10
    n_clauses = 10
    n_classes = 2
    s = 3.0
    n_literal_budget = 4
    threshold = 15    
    
    tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, n_literal_budget=n_literal_budget)

    x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, seed=41)
    tm.set_train_data(x, y)
    tm.set_test_data(ex, ey)

    trainer = gt.Trainer(threshold, n_epochs=100, seed=32, n_jobs=1, early_exit_acc=True)
    r = trainer.train(tm)    
    print("result:", r)
    
    
    
    rp = RulePredictor(True)
    fm = list(range(n_literals))

    rp.create_from_state(tm.get_state(), fm)
    
    y_hat = rp.predict(ex[0])
    y_hat, expl = rp.predict(ex[0], explain=True)
    
    print(y_hat)
    print(y_hat, expl)
    
    print(rp.get_global_importance(0))
    print(rp.get_global_importance(1))
    
            
            






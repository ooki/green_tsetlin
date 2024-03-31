
from typing import Optional, Tuple, Union, List
from collections import namedtuple

import numpy as np

import green_tsetlin as gt
from green_tsetlin.backend import impl as _backend_impl
from green_tsetlin.ruleset import RuleSet



Explanation = namedtuple('Explanation', ['literals', 'features'])

class Predictor:
    def __init__(self, explanation:str="none", exclude_negative_clauses:bool=False, multi_label:bool=False):        
        self.is_multi_label = multi_label
        self.empty_class_output = 0
        self.explanation = explanation
        self.exclude_negative_clauses = exclude_negative_clauses

        valid_explanations = set(["none", "literals", "features", "both"])
        if explanation not in valid_explanations:
            raise ValueError("explanation must be one of the following: {}".format(", ".join(valid_explanations)))
        
        self.n_literals: int = -1
        self.n_classes: int = -1
        self.n_features: int = -1

        self._predict_if_target_is_none = None
        
        self._ruleset:gt.RuleSet = None
        self.target_names:list = None
        self.inv_target_names:dict = None
        

        self._inf = None

    @staticmethod
    def from_ruleset(ruleset: RuleSet, explanation="none", exclude_negative_clauses=False) -> "Predictor":
        predictor = Predictor(explanation=explanation, exclude_negative_clauses=exclude_negative_clauses, multi_label=ruleset.is_multi_label)        
        predictor._set_ruleset(ruleset)
        predictor._allocate_backend()
        
        return predictor
    
    def export_as_program(self, to_file:str, exporter="topological_c"):
        
        if exporter == "simple_c":
            from green_tsetlin.writers.simple_c import SimpleC
            writer = SimpleC(self._ruleset)                        
            writer.to_file(to_file)
            
        elif exporter == "topological_c":
            from green_tsetlin.writers.topographical_c import TopographicalC
            writer = TopographicalC(self._ruleset)                        
            writer.to_file(to_file)                        
            
        else:
            raise ValueError("Cannot find exporter: '{}' - Unable to export Predictor.".format(exporter))
    
    def _set_ruleset(self, ruleset: RuleSet):
        self._ruleset = ruleset
        self.n_literals = self._ruleset.n_literals
        self.n_classes = self._ruleset.n_classes

    def set_features(self, feature_map: Union[List[int], np.array]):
        
        if not isinstance(feature_map, np.ndarray):
            feature_map = np.array(feature_map, dtype=np.uint32)
        else:
            feature_map = feature_map.astype(np.uint32)

        if feature_map.ndim != 1:
            raise ValueError("feature_map must be one dimensional array")

        if feature_map.shape[0] != self.n_literals:
            raise ValueError("feature_map must have the same number of elements as n_literals: {} != {}".format(feature_map.shape[0], self.n_literals))
        
        self.feature_map = feature_map
    
    def set_target_names(self, names):
        self.target_names = names
        self.inv_target_names = {v:k for k,v in enumerate(self.target_names)}
    
    
    def predict(self, x : np.array) -> Union[Union[int, str], List[Union[int, str]]]:
        self._allocate_backend()

        self._predict_if_target_is_none = self._inf.predict(x)
        predicted_class = self._predict_if_target_is_none

        if self.target_names is not None:
            predicted_class = self.target_names[predicted_class]

        return predicted_class
    
    def _get_explanation_from_backend(self):
        if self.explanation == "literals":
            return Explanation(self._inf.get_literal_importance(), None)
            
        elif self.explanation == "features":
            return Explanation(None, self._inf.get_feature_importance())
            
        elif self.explanation == "both":
            return Explanation(self._inf.get_literal_importance(), self._inf.get_feature_importance())
            
        raise ValueError("Explain: '{}' not supported.".format(self.explanation))
        
            

    def explain(self, explain="target", target=None) -> Explanation:
        if self.explanation == "none":
            raise ValueError("Cannot request explanation on a predictor that has explanation set to 'none'.")        
        self._allocate_backend()
        
        
        if explain == "target":
            if target is None:
                class_target = self._predict_if_target_is_none
            else:
                class_target = target                            
                if self.target_names is not None:
                    try:
                        class_target = self.inv_target_names[target]
                    except KeyError:
                        raise ValueError("Target must be one of the following: {}".format(", ".join(self.target_names)))
            
            self._inf.calculate_explanations(class_target)
            return self._get_explanation_from_backend()
                
        elif explain == "all":
            all_expl = {}
            for target_class in range(self.n_classes):                
                self._inf.calculate_explanations(target_class)                                
                all_expl[target_class] = self._get_explanation_from_backend()
            
            if self.target_names is not None:                
                return dict( (self.target_names[k], v) for k, v in all_expl)
       
            return all_expl
        
        else:
            raise ValueError("Method {} not supported as a target, try: 'all' or a target. (Or None for last prediction).".format(explain))
    
    def predict_and_explain(self, x:np.array):
        y_hat = self.predict(x)
        return y_hat, self.explain()
    

    def _allocate_backend(self):
        if self._inf is None:
            self._create_backend_inference()

    def _create_backend_inference(self):        
        backend_cls = self._get_backend()

        self._inf = backend_cls(self.n_literals, self.n_classes, self.n_features)        
        self._inf.set_rules(self._ruleset.rules, self._ruleset.weights)

    def _get_backend(self):
        weigth_flag = "Wt" if self.exclude_negative_clauses else "Wf"
        backend_name = ""
        if self.explanation == "none":
            backend_name = "Inference8u_Ff_Lf_Wf"

        elif self.explanation == "literals":
            backend_name = "Inference8u_Ff_Lt_{}".format(weigth_flag)

        elif self.explanation == "features":
            backend_name = "Inference8u_Ft_Lf_{}".format(weigth_flag)

        elif self.explanation == "both":
            backend_name = "Inference8u_Ft_Lt_{}".format(weigth_flag)
        
        else:
            raise ValueError("Could not find a backend inference object with explanation set to '{}'".format(self.explanation))

        return _backend_impl[backend_name]



    


    





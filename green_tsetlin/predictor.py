
from typing import Optional, Tuple, Union, List

import numpy as np

import green_tsetlin as gt
from green_tsetlin.backend import impl as _backend_impl
from green_tsetlin.ruleset import RuleSet




class Predictor:
    def __init__(self, multi_label:bool=False, explanation:str="none"):        
        self.is_multi_label = multi_label
        self.empty_class_output = 0
        self.explanation = explanation
        
        self.n_literals: int = -1
        self.n_clauses: int = -1
        self.n_classes: int = -1
        self.n_features: int = -1
        
        self._ruleset:gt.RuleSet = None
        self.explanation_names:list = None
        self.target_names:list = None

    def set_explanation_names(self, names):
        self.explanation_names = names

    
    def set_target_names(self, names):
        self.target_names = names
    
    
    def predict(self, x : np.array) -> Union[Union[int, str], List[Union[int, str]]]:

        if self.target_names is not None:
            argmax_prediction = self.target_names[argmax_prediction]

        return 0
    

    def explain(self, x : np.array) -> Tuple[Union[Union[int, str], List[Union[int, str]]], list]:
        if self.explanation == "none":
            raise ValueError("Cannot request explanation on a predictor that has explanation set to 'none'.")
        
        argmax_prediction = 0
        explantion = [[0, +3], [2, -6]]

        if self.explanation_names is not None:
            explantion = [[self.explanation_names[idx], v] for idx, v in explantion]
        
        if self.target_names is not None:
            argmax_prediction = self.target_names[argmax_prediction]

        return argmax_prediction, explantion
    

    def _create_backend_inference(self):
        pass
    


    





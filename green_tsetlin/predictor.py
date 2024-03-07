
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

        self._inf = None

    def set_explanation_names(self, names):
        self.explanation_names = names

    
    def set_target_names(self, names):
        self.target_names = names
    
    
    def predict(self, x : np.array) -> Union[Union[int, str], List[Union[int, str]]]:
        self.init()

        argmax_prediction = self._inf.predict(x)
        if self.target_names is not None:
            argmax_prediction = self.target_names[argmax_prediction]

        return 0
    

    def explain(self, x : np.array) -> Tuple[Union[Union[int, str], List[Union[int, str]]], list]:
        if self.explanation == "none":
            raise ValueError("Cannot request explanation on a predictor that has explanation set to 'none'.")        
        self.init()
        
        argmax_prediction = 0
        explantion = [[0, +3], [2, -6]]

        if self.explanation_names is not None:
            explantion = [[self.explanation_names[idx], v] for idx, v in explantion]
        
        if self.target_names is not None:
            argmax_prediction = self.target_names[argmax_prediction]

        return argmax_prediction, explantion
    

    def init(self):
        if self._inf is None:
            self._create_backend_inference()


    def _create_backend_inference(self):        
        backend_cls = None
        if self.explanation == "none":
            backend_cls = _backend_impl["Inference8u_Ff_Lf_Wf"]
        
        
        if backend_cls is None:
            raise ValueError("Could not find a backend inference object with explanation set to '{}'".format(self.explanation))
        
        self._inf = backend_cls(self.n_literals, self.n_clauses, self.n_classes, self.n_features)        
        print("create inf:", self._inf)
        self._inf.set_rules(self._ruleset.rules, self._ruleset.weights)



    


    





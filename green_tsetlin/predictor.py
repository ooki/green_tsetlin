
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

        valid_explanations = set(["none", "literals", "features", "positive_weighted_literals", "positive_weighted_features"])
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
        self.init()

        self._predict_if_target_is_none = self._inf.predict(x)
        predicted_class = self._predict_if_target_is_none

        if self.target_names is not None:
            predicted_class = self.target_names[predicted_class]

        return predicted_class

    def explain(self, explain="target", target=None):
        if self.explanation == "none":
            raise ValueError("Cannot request explanation on a predictor that has explanation set to 'none'.")        
        self.init()

        if explain == "target":
            if target is None:
                expl = self._inf.calculate_importance(self._predict_if_target_is_none)
            else:
                if self.target_names is None:
                    expl = self._inf.calculate_importance(target)
                else:
                    try:
                        target = self.inv_target_names[target]
                    except KeyError:
                        raise ValueError("Target must be one of the following: {}".format(", ".join(self.target_names)))
                    
                    expl = self._inf.calculate_importance(target)
                
            return expl

        elif explain == "all":
            all_expl = []
            for target_class in range(self.n_classes):
                expl = self._inf.calculate_importance(target_class)
                all_expl.append(expl)
            
            if self.target_names is not None:                
                return dict( (self.target_names[k], v) for k, v in all_expl)

            else:
                # stack the list of lists
                return np.stack(all_expl)            

        return expl
    
    def predict_and_explain(self, x:np.array):
        y_hat = self.predict(x)
        return y_hat, self.explain()
    

    def init(self):
        if self._inf is None:
            self._create_backend_inference()


    def _create_backend_inference(self):        
        backend_cls = self._get_backend()

        self._inf = backend_cls(self.n_literals, self.n_classes, self.n_features)        
        self._inf.set_rules(self._ruleset.rules, self._ruleset.weights)

    def _get_backend(self):
        if self.explanation == "none":
            backend_cls = _backend_impl["Inference8u_Ff_Lf_Wf"]

        elif self.explanation == "literals":
            backend_cls = _backend_impl["Inference8u_Ff_Lt_Wf"]

        elif self.explanation == "features":
            raise NotImplementedError("Not implemented backend for explanation 'features' yet!")
            #backend_cls = _backend_impl["Inference8u_Ft_Lf_Wf"]

        elif self.explanation == "positive_weighted_literals":
            backend_cls = _backend_impl["Inference8u_Ff_Lt_Wt"]

        elif self.explanation == "positive_weighted_features":
            #backend_cls = _backend_impl["Inference8u_Ft_Lf_Wt"]        
            raise NotImplementedError("Not implemented backend for explanation 'positive_weighted_features' yet!")
        
        if backend_cls is None:
            raise ValueError("Could not find a backend inference object with explanation set to '{}'".format(self.explanation))

        return backend_cls



    


    





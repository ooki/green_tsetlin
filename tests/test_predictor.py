from collections import namedtuple

import pytest
import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc




class MockRuleset:
    def __init__(self):
        self.rules = [[0], [0,1], [0,1], [2,3]]
        self.weights = [[-1, 2], [-3, 4], [5, -6], [7, -8]]
        self.n_literals = 2
        self.n_classes = 2


def test_init():
    

    p = gt.Predictor(multi_label=False, explanation="none")
    m = MockRuleset()
    p._set_ruleset(m)    
    p.init()




    #y_hat = predictor.predict(np.array([0,1,1,1]))
    #print(y_hat)

    #y_hat, expl = predictor.explain([0,1,1,1])



if __name__ == "__main__":
    test_init()
    print("<done tests:", __file__, ">")


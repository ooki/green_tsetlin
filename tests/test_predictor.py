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


class MockXorRuleset:
    def __init__(self):
        self.rules = [[1, 8], [8, 9], [0, 9]]
        self.weights = [[-21, 21], [24, -21], [-26, 28]]
        self.n_literals = 8
        self.n_classes = 2
    

def test_init():    
    p = gt.Predictor(multi_label=False, explanation="none")
    m = MockRuleset()
    p._set_ruleset(m)    
    p._allocate_backend()

    assert p.n_literals == 2
    assert p.n_classes == 2


def test_sets_correct_backend_based_on_exploration_and_enw():
    e_and_backend = [
        ("none", False, gtc.Inference8u_Ff_Lf_Wf),
        ("literals", True, gtc.Inference8u_Ff_Lt_Wt),
        ("literals", False, gtc.Inference8u_Ff_Lt_Wf),
        ("features", True, gtc.Inference8u_Ft_Lf_Wt),
        ("features", False, gtc.Inference8u_Ft_Lf_Wf),
        ("both", True, gtc.Inference8u_Ft_Lt_Wt),
        ("both", False, gtc.Inference8u_Ft_Lt_Wf),
    ]

    for explanation, enc, backend_cls in e_and_backend:
        p = gt.Predictor(explanation=explanation, exclude_negative_clauses=enc, multi_label=False)
        m = MockRuleset()
        p._set_ruleset(m)
        
        def empty_alloc():
            pass 
        p._allocate_backend = empty_alloc # since we dont set features, dont alloc anything
        
        assert p._get_backend() == backend_cls
    

def test_prediction_with_target_names():
    p = gt.Predictor(multi_label=False, explanation="literals")
    p.set_target_names(["a", "b"])
    assert p.target_names == ["a", "b"]
    
    m = MockRuleset()
    p._set_ruleset(m)    
    p._allocate_backend()

    assert p.predict(np.array([0,0], dtype=np.uint8)) == "a"
    assert p.predict(np.array([1,0], dtype=np.uint8)) == "b"


def test_prediction_clauses_literal_explanation():
    
    for enc, single, all in [(False, [2, 0, 0, 0], [[-1, 0, 0, 0], [2, 0, 0, 0]]),
                             (True, [2, 0, 0, 0],  [ [0, 0, 0, 0], [2, 0, 0, 0]])]:
        p = gt.Predictor(multi_label=False, explanation="literals", exclude_negative_clauses=enc)    
        m = MockRuleset()
        p._set_ruleset(m)    
        p._allocate_backend()

        x = np.array([1,0], dtype=np.uint8)
        p.predict(x)
        
        assert np.array_equal(p.explain(explain="target").literals, single)
        e_all = p.explain(explain="all")
        assert np.array_equal(e_all[0].literals, all[0])
        assert np.array_equal(e_all[1].literals, all[1])



def test_predictor_pass_xor():
    n_literals = 8
    n_clauses = 50
    n_classes = 2
    s = 3.0
    threshold = 42
        
    found_correct_tm = False
    for _ in range(10):
        tm = gt.TsetlinMachine(n_literals=n_literals, n_clauses=n_clauses, n_classes=n_classes, s=s, threshold=threshold, literal_budget=4)            
        x, y, ex, ey = gt.dataset_generator.xor_dataset(n_literals=n_literals, n_train=400, n_test=200)
        trainer = gt.Trainer(tm, seed=32, n_jobs=1, n_epochs=100, progress_bar=False)
        trainer.set_train_data(x, y)
        trainer.set_eval_data(ex, ey)
        trainer.train()
        
        if trainer.results["did_early_exit"]:
            found_correct_tm = True
            break
        
    assert found_correct_tm
    
    correct = 0
    total = 0
    p = tm.get_predictor()
    for k, xk in enumerate(ex):
        y_hat = p.predict(xk)
        if y_hat == ey[k]:
            correct += 1
            
        total += 1
    
    
    assert correct == total
        
    # if trainer.results["did_early_exit"]:           
    #     print("total:", total, "correct:", correct)
        
    #     if total != correct:
        
    #         print(tm.get_ruleset().rules)
    #         print(tm.get_ruleset().weights)
            
    #         print("xk:", wrong_example.tolist(), "y:", wrong_example_y, "pred:", wrong_example_pred)
    #     else:
    #         print("all good.")
      
            
    
# def test_mnist_predict():
#     from sklearn.datasets import fetch_openml
#     X, y = fetch_openml(
#             "mnist_784",
#             version=1,
#             return_X_y=True,
#             as_frame=False)
    
    
    
    
    
def test_xor_predict():     
    p = gt.Predictor()    
    m = MockXorRuleset()
    p._set_ruleset(m)    
    p._allocate_backend()
    
    for r, w in zip(m.rules, m.weights):
        print("rules:", r,  " w:", w)

    y_hat = p.predict(np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=np.uint8))
    print("y_hat:", y_hat)



if __name__ == "__main__":
    # test_init()
    # test_sets_correct_backend_based_on_exploration_and_enw()
    # test_prediction_with_target_names()
    # test_prediction_literal_explanation()
    # test_prediction_clauses_literal_explanation()
    test_predictor_pass_xor()
    # test_xor_predict()
    print("<done tests:", __file__, ">")


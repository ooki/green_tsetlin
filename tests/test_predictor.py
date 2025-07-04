from collections import namedtuple

import pytest
import numpy as np

import green_tsetlin as gt
import green_tsetlin_core as gtc
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.utils import shuffle
import datasets


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


# def test_predictor_pass_mnist():

#     X, y = fetch_openml(
#                 "mnist_784",
#                 version=1,
#                 return_X_y=True,
#                 as_frame=False)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#     X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
#     X_train = X_train.astype(np.uint8)
        
#     X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
#     X_test = X_test.astype(np.uint8)
    
#     y_train = y_train.astype(np.uint32)
#     y_test = y_test.astype(np.uint32)

#     n_literals = X_train.shape[1]
#     n_clauses = 500
#     n_classes = 10
#     s = 10.0
#     threshold = 625
    

#     tm = gt.TsetlinMachine(n_literals=n_literals,
#                            n_clauses=n_clauses,
#                            n_classes=n_classes,
#                            s=s,
#                            threshold=threshold
#                            )
    
#     trainer = gt.Trainer(tm, seed=31, n_jobs=1, n_epochs=10, feedback_type='uniform', progress_bar=False)

#     trainer.set_train_data(X_train, y_train)

#     trainer.set_eval_data(X_test, y_test)

#     r = trainer.train()

#     predictor = tm.get_predictor()

#     y_pred = np.zeros(y_test.shape)
#     for i in range(len(X_test)):
        
#         y_pred[i] = predictor.predict(X_test[i])

#     predictor_score = np.mean(y_pred==y_test)

#     assert predictor_score == r['best_eval_score']


# def test_predictor_pass_trec():

#     trec = datasets.load_dataset('trec')
    
#     x_train, y_train, x_test, y_test = trec['train']['text'], trec['train']['coarse_label'], trec['test']['text'], trec['test']['coarse_label']

#     x_train, y_train = shuffle(x_train, y_train, random_state=42)

#     x_test, y_test = shuffle(x_test, y_test, random_state=42)

#     vectorizer = CountVectorizer(
#         analyzer = 'word',
#         binary=True,
#         ngram_range=(1, 3),
#         max_features=100_000,
#         max_df=0.8,
#         min_df=3,
#     )


#     x_train = vectorizer.fit_transform(x_train).toarray().astype(np.uint8)
#     y_train = np.array(y_train).astype(np.uint32)
#     x_test = vectorizer.transform(x_test).toarray().astype(np.uint8)
#     y_test = np.array(y_test).astype(np.uint32)

    
#     SKB = SelectKBest(score_func=chi2, k=784)
#     SKB.fit(x_train, y_train)

#     X_train = SKB.transform(x_train)
#     X_test = SKB.transform(x_test)


#     n_literals = X_train.shape[1]
#     n_clauses = 500
#     n_classes = 6
#     s = 10.0
#     threshold = 625
    

#     tm = gt.TsetlinMachine(n_literals=n_literals,
#                            n_clauses=n_clauses,
#                            n_classes=n_classes,
#                            s=s,
#                            threshold=threshold
#                            )
    
#     trainer = gt.Trainer(tm, seed=31, n_jobs=1, n_epochs=10, progress_bar=False)

#     trainer.set_train_data(X_train, y_train)

#     trainer.set_eval_data(X_test, y_test)

#     r = trainer.train()

#     predictor = tm.get_predictor()

#     y_pred = np.zeros(y_test.shape)
#     for i in range(len(X_test)):
        
#         y_pred[i] = predictor.predict(X_test[i])

#     predictor_score = np.mean(y_pred==y_test)

#     assert predictor_score == r['best_eval_score']




if __name__ == "__main__":
    # test_init()
    # test_sets_correct_backend_based_on_exploration_and_enw()
    # test_prediction_with_target_names()
    # test_prediction_literal_explanation()
    # test_prediction_clauses_literal_explanation()
    # test_predictor_pass_xor()
    # test_xor_predict()
    print("<done tests:", __file__, ">")


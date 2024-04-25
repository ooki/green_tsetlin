

import numpy as np

from green_tsetlin.hpsearch import HyperparameterSearch
from green_tsetlin.dataset_generator import xor_dataset


def test_set_data():
    
    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.05)

    hyperparam_search = HyperparameterSearch(s_space=(2.0, 10.0),
                                              clause_space=(10, 50),
                                              threshold_space=(10, 100),
                                              max_epoch_per_trial=(10, 100),
                                              literal_budget=(5, 20),
                                              seed=42,
                                              n_jobs=5)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    assert np.array_equal(hyperparam_search.x_train, train_x)
    assert np.array_equal(hyperparam_search.y_train, train_y)
    assert np.array_equal(hyperparam_search.x_eval, test_x)
    assert np.array_equal(hyperparam_search.y_eval, test_y)


def test_trial():
    
    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.05)

    hyperparam_search = HyperparameterSearch(s_space=(2.0, 10.0),
                                              clause_space=(10, 50),
                                              threshold_space=(10, 100),
                                              max_epoch_per_trial=(10, 100),
                                              literal_budget=(5, 20),
                                              seed=42,
                                              n_jobs=5)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    class MockTrial:
        def suggest_float(self, name, low, high):
            return (low + high) / 2
        
        def suggest_int(self, name, low, high):
            return (low + high) // 2

    trial = MockTrial()
    
    hyperparam_search.objective(trial)


def test_optimization_literals():

    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.05)

    hyperparam_search = HyperparameterSearch(s_space=(2.0, 20.0),
                                              clause_space=(5, 10),
                                              threshold_space=(3, 20),
                                              max_epoch_per_trial=10,
                                              literal_budget=(1, train_x.shape[1]),
                                              minimize_literal_budget=True,
                                              seed=42,
                                              n_jobs=5)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    vals = []
    literal_max = 3
    
    for i in range(5):

        hyperparam_search.optimize(n_trials=10, study_name="none", show_progress_bar=True)
        
        for trial in hyperparam_search.best_trials:
            vals.append(trial.values)

    vals = np.array(vals)

    assert np.min(vals[:, 1]) < literal_max, f"mean literal budget should be less than {literal_max}, got literal values {vals[:, 1]}"


def test_set_params():

    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.05)

    hyperparam_search = HyperparameterSearch(s_space=5.0,
                                              clause_space=50,
                                              threshold_space=30,
                                              max_epoch_per_trial=5, 
                                              literal_budget=5,
                                              seed=42,
                                              n_jobs=5)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    hyperparam_search.optimize(n_trials=1, study_name="none")

    for trial in hyperparam_search.best_trials:
        assert trial.params == {}
    
    hyperparam_search = HyperparameterSearch(s_space=5.0,
                                              clause_space=(50, 100),
                                              threshold_space=30,
                                              max_epoch_per_trial=5, 
                                              literal_budget=5,
                                              seed=42,
                                              n_jobs=5)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    hyperparam_search.optimize(n_trials=1, study_name="none")

    for trial in hyperparam_search.best_trials:
        assert list(trial.params.keys()) == ["n_clauses"]


def test_optimization():

    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.1)

    hyperparam_search = HyperparameterSearch(s_space=(2.0, 20.0),
                                              clause_space=(5, 10),
                                              threshold_space=(3, 20),
                                              max_epoch_per_trial=20,
                                              literal_budget=(1, train_x.shape[1]),
                                              seed=42,
                                              n_jobs=5,
                                              k_folds=4,
                                              minimize_literal_budget=False)
    
    hyperparam_search.set_train_data(train_x, train_y)
    hyperparam_search.set_eval_data(test_x, test_y)

    hyperparam_search.optimize(n_trials=10, study_name="none", show_progress_bar=True)


def test_with_kfold():

    train_x, train_y, test_x, test_y = xor_dataset(n_train=50, n_test=10, n_literals=8, seed=42, noise=0.05)

    vals = []
    
    for i in range(5):

        seed = np.random.randint(1, 1000)

        hyperparam_search = HyperparameterSearch(s_space=(2.0, 20.0),
                                                clause_space=(5, 20),
                                                threshold_space=(3, 20),
                                                max_epoch_per_trial=20,
                                                literal_budget=(1, train_x.shape[1]),
                                                seed=seed,
                                                n_jobs=5,
                                                k_folds=5)
        
        hyperparam_search.set_train_data(train_x, train_y)
        hyperparam_search.set_eval_data(test_x, test_y)

        hyperparam_search.optimize(n_trials=3, study_name="none", show_progress_bar=True)
        
        for trial in hyperparam_search.best_trials:
            vals.append(trial.values)

    assert np.max(vals) == 1.0



if __name__ == "__main__":
    
    # test_optimization_literals()
    # test_set_params()
    # test_trial()
    # test_set_data()
    # test_optimization()
    # test_with_kfold()

    print("<done tests:", __file__, ">")
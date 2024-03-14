from typing import Tuple, Union
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import datasets
import optuna

from green_tsetlin.tsetlin_machine import TsetlinMachine
from green_tsetlin.trainer import Trainer


class HyperparameterSearch:

    def __init__(self, 
                 s_space : Union[Tuple[float, float], float], 
                 clause_space : Union[Tuple[int, int], int], 
                 threshold_space : Union[Tuple[int, int], int], 
                 max_epoch_per_trial : Union[Tuple[int, int], int],
                 literal_budget : Union[Tuple[int, int], int],
                 minimize_literal_budget : bool = False,
                 seed : int = 42, 
                 n_jobs : int = -1):
        
        self.s_space = s_space
        self.clause_space = clause_space
        self.threshold_space = threshold_space
        self.seed = seed
        self.max_epoch_per_trial = max_epoch_per_trial
        self.n_jobs = n_jobs
        self.literal_budget = literal_budget
        self.minimize_literal_budget = minimize_literal_budget


    def set_train_data(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y


    def set_test_data(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y


    def _check_data(self):

        if self.train_x is None:
            raise ValueError("Train data not set. Use set_train_data()")
        
        elif self.test_x is None:
            raise ValueError("Validation data not set. Use set_validation_data()")


    def objective(self, trial):
        
        self._check_data()

        s = trial.suggest_float("s", self.s_space[0], self.s_space[1]) if not isinstance(self.s_space, float) else self.s_space
        clauses = trial.suggest_int("n_clauses", self.clause_space[0], self.clause_space[1]) if not isinstance(self.clause_space, int) else self.clause_space
        threshold = trial.suggest_float("threshold", self.threshold_space[0], self.threshold_space[1]) if not isinstance(self.threshold_space, int) else self.threshold_space
        literal_budget = trial.suggest_int("literal_budget", self.literal_budget[0], self.literal_budget[1]) if not isinstance(self.literal_budget, int) else self.literal_budget
        max_epoch_per_trial = trial.suggest_int("max_epoch_per_trial", self.max_epoch_per_trial[0], self.max_epoch_per_trial[1]) if not isinstance(self.max_epoch_per_trial, int) else self.max_epoch_per_trial

        tm = TsetlinMachine(n_literals=self.train_x.shape[1], 
                            n_clauses=clauses, 
                            s=s,
                            threshold=threshold,
                            n_classes=len(np.unique(self.train_y)),
                            literal_budget=literal_budget)

        trainer = Trainer(tm=tm, 
                          n_jobs=self.n_jobs,
                          n_epochs=max_epoch_per_trial,
                          seed=self.seed,
                          progress_bar=False)
        
        trainer.set_train_data(self.train_x, self.train_y)
        trainer.set_test_data(self.test_x, self.test_y)

        trainer.train()
        
        res = trainer.results["best_test_score"]

        if self.minimize_literal_budget:
            return res, tm.literal_budgets[0]

        return res

    def optimize(self, n_trials, study_name, storage : str = None, return_best : bool = False, show_progress_bar : bool = False):
        
        if self.minimize_literal_budget:
            study = optuna.create_study(study_name=study_name, storage=storage, directions=["maximize", "minimize"], load_if_exists=True)
        
        else:
            study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
        
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        if return_best:
            return study.best_trials



if __name__ == "__main__":

    s_space = (2.0, 10.0)
    clause_space = (100, 500)
    threshold_space = (50, 1000)
    seed = 42
    epochs = 20
    n_jobs = 1

    rng = np.random.default_rng(seed)

    tm_hp = HyperparameterSearch(s_space, clause_space, threshold_space, seed, epochs, n_jobs)
    
    imdb = datasets.load_dataset('imdb')
    x, y = imdb['train']['text'], imdb['train']['label']
    
    vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, lowercase=True, max_features=5000)
    x_bin = vectorizer.fit_transform(x).toarray()
    x_bin = x_bin.astype(np.uint8)
    y = np.array(y).astype(np.uint32)
    
    shuffle_index = [i for i in range(len(x))]
    rng.shuffle(shuffle_index)

    x_bin = x_bin[shuffle_index]
    y = y[shuffle_index]

    x_bin = x_bin[:1000]
    y = y[:1000]

    train_x_bin, val_x_bin, train_y, val_y = train_test_split(x_bin, y, test_size=0.2, random_state=seed, shuffle=True)

    tm_hp.set_train_data(train_x_bin, train_y)
    tm_hp.set_validation_data(val_x_bin, val_y)

    best_score = tm_hp.optimize(n_trials=2, study_name="IMDB_study", storage=None, return_best=True)

    print(best_score)






    
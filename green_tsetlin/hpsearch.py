from typing import Tuple, Union

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import optuna

from green_tsetlin.tsetlin_machine import TsetlinMachine
from green_tsetlin.trainer import Trainer


class HyperparameterSearch:

    """
    Example
    --------

    .. code-block:: python

        from green_tsetlin.hpsearch import HyperparameterSearch
        from green_tsetlin.dataset_generator import xor_dataset

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
        hyperparam_search.set_test_data(test_x, test_y)

        hyperparam_search.optimize(trials=10, 
                                   study_name="xor study", 
                                   show_progress_bar=True,
                                   storage="sqlite:///xor.db")
    
    """

    def __init__(self, 
                 s_space : Union[Tuple[float, float], float], 
                 clause_space : Union[Tuple[int, int], int], 
                 threshold_space : Union[Tuple[int, int], int], 
                 max_epoch_per_trial : Union[Tuple[int, int], int],
                 literal_budget : Union[Tuple[int, int], int],
                 minimize_literal_budget : bool = False,
                 seed : int = 42, 
                 n_jobs : int = -1,
                 k_folds : int = 0):
        
        self.s_space = s_space
        self.clause_space = clause_space
        self.threshold_space = threshold_space
        self.seed = seed
        self.max_epoch_per_trial = max_epoch_per_trial
        self.n_jobs = n_jobs
        self.literal_budget = literal_budget
        self.minimize_literal_budget = minimize_literal_budget
        self.k_folds = k_folds


    def set_train_data(self, x_train, y_train):
        """
        Set the training data for the study.

        Parameters:
            train_x (array): The input training data.
            train_y (array): The target training data.

        """
        
        self.x_train = x_train
        self.y_train = y_train


    def set_eval_data(self, x_eval, y_eval):
        """
        Set the test data for the model.

        Parameters:
            test_x (array): The input test data.
            test_y (array): The target test data.

        """

        self.x_eval = x_eval
        self.y_eval = y_eval


    def _check_data(self):

        if self.x_eval is None:
            raise ValueError("Train data not set. Use set_train_data()")
        
        elif self.x_train is None:
            raise ValueError("Validation data not set. Use set_validation_data()")


    def objective(self, trial):
        """
        Parameters:
            trial (optuna.trial.Trial): A trial object used to generate parameters for optimization.
        
        Returns:
            float: The best test score achieved during training.
            int: The literal budget used for optimization if specified.
        """

        self._check_data()

        s = trial.suggest_float("s", self.s_space[0], self.s_space[1]) if not isinstance(self.s_space, float) else self.s_space
        clauses = trial.suggest_int("n_clauses", self.clause_space[0], self.clause_space[1]) if not isinstance(self.clause_space, int) else self.clause_space
        threshold = trial.suggest_float("threshold", self.threshold_space[0], self.threshold_space[1]) if not isinstance(self.threshold_space, int) else self.threshold_space
        literal_budget = trial.suggest_int("literal_budget", self.literal_budget[0], self.literal_budget[1]) if not isinstance(self.literal_budget, int) else self.literal_budget
        max_epoch_per_trial = trial.suggest_int("max_epoch_per_trial", self.max_epoch_per_trial[0], self.max_epoch_per_trial[1]) if not isinstance(self.max_epoch_per_trial, int) else self.max_epoch_per_trial

        tm = TsetlinMachine(n_literals=self.x_train.shape[1], 
                            n_clauses=clauses, 
                            s=s,
                            threshold=threshold,
                            n_classes=len(np.unique(self.y_train)),
                            literal_budget=literal_budget)

        trainer = Trainer(tm=tm, 
                          n_jobs=self.n_jobs,
                          n_epochs=max_epoch_per_trial,
                          seed=self.seed,
                          progress_bar=False,
                          k_folds=self.k_folds,
                          kfold_progress_bar=False)
        
        trainer.set_train_data(self.x_train, self.y_train)
        trainer.set_eval_data(self.x_eval, self.y_eval)

        res = trainer.train()
        
        res = trainer.results["best_eval_score"]

        if self.minimize_literal_budget:
            return res, tm.literal_budgets[0]

        return res


    def optimize(self, n_trials, study_name, storage : str = None, show_progress_bar : bool = False):
        
        """
        A method to optimize the parameters using Optuna for a given number of trials.
        
        Parameters:
            n_trials (int): The number of trials for optimization.
            study_name (str): The name of the study.
            storage (str, optional): The storage name. Defaults to None.
            show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to False.
            
        """


        if self.minimize_literal_budget:
            study = optuna.create_study(study_name=study_name, storage=storage, directions=["maximize", "minimize"], load_if_exists=True)
        
        else:
            study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
        
        # study.optimize(self.objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        with tqdm(total=n_trials, disable=show_progress_bar is False) as bar:
            
            bar.set_description("Processing trial 1 of {}, best score: NA".format(n_trials))

            for i in range(n_trials):
                
                trial = study.ask()
                value = self.objective(trial)
                study.tell(trial, value)

                bar.set_description("Processing trial {} of {}, best score: {}".format(i, n_trials, study.best_trials[0].values))
                bar.update(1)

        self.best_trials = study.best_trials



if __name__ == "__main__":
    import datasets
    
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
    tm_hp.set_eval_data(val_x_bin, val_y)

    tm_hp.optimize(n_trials=2, study_name="IMDB_study", storage=None, return_best=True)

    print(tm_hp.best_trails)








    
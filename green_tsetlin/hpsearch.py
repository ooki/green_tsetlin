from typing import TypeAlias, Tuple, Union

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
        hyperparam_search.set_eval_data(test_x, test_y)

        hyperparam_search.optimize(trials=10, 
                                   study_name="xor study", 
                                   show_progress_bar=True,
                                   storage="sqlite:///xor.db")
    
    """

    def __init__(self, 
                 s_space : Union[Tuple[float, float, float], Tuple[float, float], float], 
                 clause_space : Union[Tuple[int, int, int], Tuple[int, int], int], 
                 threshold_space : Union[Tuple[int, int, int], Tuple[int, int], int], 
                 max_epoch_per_trial : Union[Tuple[int, int, int], Tuple[int, int], int],
                 literal_budget : Union[Tuple[int, int, int], Tuple[int, int], int],
                 search_or_use_boost_true_positives: Tuple[bool, bool], 
                 minimize_literal_budget : bool = False,
                 seed : int = 42, 
                 n_jobs : int = 1,
                 k_folds : int = 1):
        
        """
        Initialize the hyper-parameter search space and general training options for a
        Tsetlin Machine (TM).

        Parameters
        ----------
        s_space : Union[Tuple[float, float, float], Tuple[float, float], float]
            Search space for the `s` parameter.

            * ``(low, high, step)`` – scan the inclusive range ``[low, high]`` with the given step size.  
            * ``(low, high)``       – scan every integer value in the inclusive range ``[low, high]``.  
            * ``value``             – use a single fixed value.

        clause_space : Union[Tuple[int, int, int], Tuple[int, int], int]
            Search space for the number of clauses, interpreted in the same way
            as *s_space*.

        threshold_space : Union[Tuple[int, int, int], Tuple[int, int], int]
            Search space for the threshold parameter, interpreted as above.

        max_epoch_per_trial : Union[Tuple[int, int, int], Tuple[int, int], int]
            Search space for the maximum number of training epochs **per trial**.

        literal_budget : Union[Tuple[int, int, int], Tuple[int, int], int]
            Search space for the literal budget (upper bound on literals per clause).

        search_or_use_boost_true_positives: Tuple[bool, bool]
            First bool is if you want to search, other one is if only use or not use.

        minimize_literal_budget : bool, default ``False``
            If ``True`` the tuner tries to **minimise** the literal budget;
            otherwise it treats larger budgets as potentially beneficial.

        seed : int, default ``42``
            Random seed used for all stochastic components.

        n_jobs : int, default ``1``
            Number of parallel jobs (processes) when fitting multiple TMs.

        k_folds : int, default ``1``
            Number of folds for cross-validation during each hyper-parameter
            evaluation. A value of ``1`` disables cross-validation.

        Notes
        -----
        *Any* of the ``*_space`` arguments may be supplied in any of the three
        formats shown above.  Using a three-tuple enables non-unit step sizes
        (e.g. ``(1, 10, 2) → 1, 3, 5, 7, 9``).  Two-tuple inputs default to a
        step of ``1``; single values fix the parameter.
        """
        
        self.s_space = s_space
        self.clause_space = clause_space
        self.threshold_space = threshold_space
        self.seed = seed
        self.max_epoch_per_trial = max_epoch_per_trial
        self.n_jobs = n_jobs
        self.literal_budget = literal_budget
        self.minimize_literal_budget = minimize_literal_budget
        self.search_or_use_boost_true_positives = search_or_use_boost_true_positives
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
            raise ValueError("validation data not set. Use set_train_data()")
        
        elif self.x_train is None:
            raise ValueError("train data not set. Use set_eval_data()")


    def objective(self, trial):
        """
        Parameters:
            trial (optuna.trial.Trial): A trial object used to generate parameters for optimization.
        
        Returns:
            float: The best test score achieved during training.
            int: The literal budget used for optimization if specified.
        """

        self._check_data()

        s = (
            self.s_space
            if isinstance(self.s_space, float)               # fixed value → use as-is
            else trial.suggest_float(
                "s",
                self.s_space[0],
                self.s_space[1],
                step=self.s_space[2] if len(self.s_space) == 3 else 0.1   # default = 0.1
            )
        )

        clauses = (
            self.clause_space
            if isinstance(self.clause_space, int)
            else trial.suggest_int(
                "n_clauses",
                self.clause_space[0],
                self.clause_space[1],
                step=self.clause_space[2] if len(self.clause_space) == 3 else 1   # default = 1
            )
        )

        threshold = (
            self.threshold_space
            if isinstance(self.threshold_space, int)
            else trial.suggest_int(
                "threshold",
                self.threshold_space[0],
                self.threshold_space[1],
                step=self.threshold_space[2] if len(self.threshold_space) == 3 else 1
            )
        )

        literal_budget = (
            self.literal_budget
            if isinstance(self.literal_budget, int)
            else trial.suggest_int(
                "literal_budget",
                self.literal_budget[0],
                self.literal_budget[1],
                step=self.literal_budget[2] if len(self.literal_budget) == 3 else 1
            )
        )

        max_epoch_per_trial = (
            self.max_epoch_per_trial
            if isinstance(self.max_epoch_per_trial, int)
            else trial.suggest_int(
                "max_epoch_per_trial",
                self.max_epoch_per_trial[0],
                self.max_epoch_per_trial[1],
                step=self.max_epoch_per_trial[2] if len(self.max_epoch_per_trial) == 3 else 1
            )
        )

        if self.search_or_use_boost_true_positives[0]:
            boost_true_positives = trial.suggest_int("boost_true_positives", 0, 1)
        else:
            boost_true_positives = self.search_or_use_boost_true_positives[1]

        tm = TsetlinMachine(n_literals=self.x_train.shape[1], 
                            n_clauses=clauses, 
                            s=s,
                            threshold=threshold,
                            n_classes=len(np.unique(self.y_train)),
                            literal_budget=literal_budget,
                            boost_true_positives=boost_true_positives)

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
        
        res = trainer.results["eval_log"][-1]

        if self.minimize_literal_budget:
            return res, tm.literal_budgets[0]

        return res


    def optimize(self, n_trials, study_name, storage : str = None, show_progress_bar : bool = True):
        
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
        
        with tqdm(total=n_trials, disable=show_progress_bar is False) as bar:
            
            bar.set_description("Processing trial 1 of {}, best score: NA".format(n_trials))

            for i in range(n_trials):
                
                trial = study.ask()
                value = self.objective(trial)
                study.tell(trial, value)

                bar.set_description("Processing trial {} of {}, best score: {}".format(i+1, n_trials, study.best_trials[0].values))
                bar.update(1)

        self.best_trials = study.best_trials
        for t in self.best_trials:
            print(t.params)



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








    

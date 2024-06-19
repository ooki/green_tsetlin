IMDB : highly polar text sentiment classification
==================================================

Here we show green\_tsetlin Tsetlin Machine trains on the **IMDB sentiment dataset**.  

.. code-block:: python

    import datasets

    imdb = datasets.load_dataset('imdb')
    x, y = imdb['train']['text'], imdb['train']['label']

We can vectorize the text data using sklearn `CountVectorizer`. 
This lets us convert text data to a **sparse matrix**. 

.. code-block:: python

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True, lowercase=True, max_features=5_000)
    vectorizer.fit(x)

green\_tsetlin is compatible with **sparse data**. 
As the `CountVectorizer` returns a sparse matrix, 
we can either choose to use the sparse data as it is 
or convert it to dense data. Other options is 
using `gt.SparseTsetlinMachine` that handles sparse data as sparse.

.. code-block:: python

    import numpy as np

    x_bin = vectorizer.transform(x).toarray().astype(np.uint8)
    y = np.array(y).astype(np.uint32)

With sklearn `train_test_split` 
we can split the data into train and validation sets.

.. code-block:: python

    from sklearn.model_selection import train_test_split as split

    train_x_bin, val_x_bin, train_y, val_y = split(x_bin, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True)


Install the **green-tsetlin** package using **pip**.

.. code-block:: bash

    pip install green-tsetlin


With a number of different parameters to set in the TM, we can optimize by using the built in TM optuna optimizer, `gt.hpsearch.HyperparameterSearch`.

HyperparameterSearch:

- **search spaces**: Set a disired search space for each paramater. Either set the search space to a tuple, e.g (1, 4) will search between 1 and 4, or set it to a single value. E.g 4 will only search on 4. `clause_space=(50, 250)` or `clause_space=125` 

- **literal budget**: Optimize for a minimum literal budget by setting `minimize_literal_budget=True`.

- **Cross validation**: Set `k_folds=k` to an integer $k > 2$ to run cross validation k times on each trial

HyperparameterSearch.optimize:

- Run optimization over `n_trials`, store in database, e.g `"sqlite:///my_database.db"`. 

`See the Optuna documentation here <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html>`_

.. code-block:: python

    from green_tsetlin.hpsearch import HyperparameterSearch

    hpsearch = HyperparameterSearch(s_space=(2.0, 20.0),
                                    clause_space=(100, 1000),
                                    threshold_space=(100, 1500),
                                    max_epoch_per_trial=3,
                                    literal_budget=(5, 10),
                                    k_folds=1,
                                    n_jobs=5,
                                    seed=42,
                                    minimize_literal_budget=False)

    hpsearch.set_train_data(train_x_bin, train_y)
    hpsearch.set_eval_data(val_x_bin, val_y)

    hpsearch.optimize(n_trials=1, 
                    study_name="IMDB hpsearch", 
                    show_progress_bar=True, 
                    storage=None)

We get the best hyperparameters:

.. code-block:: python

    params = hpsearch.best_trials[0].params
    performance = hpsearch.best_trials[0].values

    print("best paramaters: ", params)
    print("best score: ", performance)


**Using the trained TM for inference** lets us predict and explain the prediction. 
This means, given a set of features, we can see which features 
was important for that specific prediction.

First we have to get the predictor class. We can get explanations on literals, features or both.


.. code-block:: python
    
    predictor = tm.get_predictor(explanation="literals", exclude_negative_clauses=False)


Then, we want to test on a simple example:

.. code-block:: python

    example = "I thought this was a great movie, however the popcorn was bad."
    
This is not on TM format, so we need to convert it binary. This is done with the previuosly used CountVectorizer. 

**Important** : the exact same vocabulary from the CountVectorizer used to transform the data into bag of words needs to be used.

.. code-block:: python
    
    import pickle
    
    feature_names = pickle.load(open("feature_names_imdb.pkl", "rb"))
    vectorizer = CountVectorizer(vocabulary=feature_names, binary=True)


.. code-block:: python
    
    example = vectorizer.transform([example]).toarray().astype(np.uint8)


We can now proceed to predict and explain the examples:


.. code-block:: python

    pred, expl = predictor.predict_and_explain(example1)


Showing the explanation gives on insight in what features were important.

.. code-block:: python

    feature_idx = np.where(example[0] == 1)[0]
    feature_names = vectorizer.get_feature_names_out()
    feature_names = [feature_names[i] for i in feature_idx]
    explanation = explanation[0][weight_idx]
    for w, f in zip(explanation, feature_names):
        print(f"{f} : {w}")


.. code-block:: none

    bad : -75
    great : 194
    however : 0
    movie : 0
    popcorn : 0
    the : 0
    this : 0
    thought : 0
    was : 0
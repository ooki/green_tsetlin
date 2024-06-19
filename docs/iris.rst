Iris : continuous features    
===========================

Here we show green\_tsetlin Tsetlin Machine trains on the **Iris dataset**.

.. code-block:: python

    from sklearn.datasets import load_iris

We start by loading the Iris dataset from sklearn:

.. code-block:: python

    iris = load_iris()

    x = iris['data'].astype(np.uint8)
    y = iris['target'].astype(np.uint32)

As features of this dataset is continuous, we 
will need to convert it to TM friendly binary data.

.. code-block:: python

    import numpy as np

    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)

    intervals = (x_max - x_min) / 4

    intervals_list = [[x_min[i] + k * intervals[i] for k in range(4)] for i in range(x.shape[1])]

    x_empty = np.zeros((x.shape[0], x.shape[1] * 4)).astype(np.uint8)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_empty[i, j*4:(j+1)*4] = np.array([1 if x[i, j] >= intervals_list[j][k] else 0 for k in range(4)])

    x = x_empty

Split the data into train and test:

.. code-block:: python

    from sklearn.model_selection import train_test_split as split

    train_x, val_x, train_y, val_y = split(x, y, test_size=0.2, random_state=42, shuffle=True)


With the built in TM optuna optimizer, `gt.hpsearch.HyperparameterSearch` 
we can optimize the hyperparameters of the Tsetlin Machine.

.. code-block:: python

    from green_tsetlin.hpsearch import HyperparameterSearch

    hpsearch = HyperparameterSearch(s_space=(2.0, 30.0),
                                    clause_space=(100, 1000),
                                    threshold_space=(50, 1500),
                                    max_epoch_per_trial=30,
                                    literal_budget=(5, 10),
                                    k_folds=4,
                                    n_jobs=5,
                                    seed=42,
                                    minimize_literal_budget=False)

    hpsearch.set_train_data(train_x, train_y)
    hpsearch.set_eval_data(val_x, val_y)

    hpsearch.optimize(n_trials=10, 
                    study_name="IRIS hpsearch", 
                    show_progress_bar=True, 
                    storage=None)

We get the best hyperparameters:

.. code-block:: python

    params = hpsearch.best_trials[0].params
    performance = hpsearch.best_trials[0].values

    print("best paramaters: ", params)
    print("best score: ", performance)
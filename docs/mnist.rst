MNIST : 28*28 grayscale image multi-class classification
=========================================================

In this tutorial we show how green\_tsetlin TM can be used to train on the **MNIST dataset**. MNIST is a benchmark by digit recognition 
that contains images of handwritten digits with a total of 70,000 images. Each image is a 28x28 pixel grayscale image with values between 0 and 255.

.. code-block:: python

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split as split

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42, shuffle=True)
    

.. math::

    (70000, 784) \leftarrow (70000, 28, 28)

With sklearn we import an easy to use version of MNIST. This version gives 2d right away hence no flatten is needed. Next, as the 
TM requires binary values, each pixel is converted with a threshold of 75.

.. code-block:: python
    
    X_train = np.where(X_train > 75, 1, 0)
    X_train = X_train.astype(np.uint8)
        
    X_test = np.where(X_test > 75, 1, 0)
    X_test = X_test.astype(np.uint8)

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)


We can now train the Tsetlin Machine. Here, it is preferable and recommended to run a hyperparameter search. Head to the IMDB or IRIS tutorial to see how. Parameters set
here is from a previous search.

.. code-block:: python

    import green_tsetlin as gt

    tm = gt.TsetlinMachine(n_literals=28*28, 
                           n_clauses=6154,
                           n_classes=10,
                           s=21.627727185060525, 
                           threshold=1218)

    trainer = gt.Trainer(tm, seed=42, n_epochs=10, n_jobs=1, k_folds=2)

    trainer.set_train_data(X_train, y_train)
    trainer.set_eval_data(X_test, y_test)

    trainer.train()

    results = trainer.results


.. code-block:: python

    print(results)


.. code-block:: none 

    {
    'best_eval_score': 0.9918857142857143,
    'k_folds': 2,
    'train_time_of_epochs': 
    [41.99848390498664,
    30.74028508097399,
    30.064865624008235,
    ...
    27.486269582004752,
    27.345113909977954,
    26.679121892957482],
    'train_log': 
    [0.8924285714285715,
    0.9485714285714286,
    0.9587142857142857,
    ...
    0.9949428571428571,
    0.9950571428571429,
    0.9953714285714286],
    'eval_log': 
    [0.9393714285714285,
    0.9510857142857143,
    0.9575428571428571,
    ...
    0.9827714285714285,
    0.9822285714285715,
    0.9806285714285714]
    }
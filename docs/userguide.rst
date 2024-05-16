How to use Green Tsetlin
========================

Basic Usage
-------------
With green\_tsetlin built in dependencies, few packages is required other than green\_tsetlin.

.. code-block:: python

    import green_tsetlin as gt

Get a dataset that has binary features. For this simple example, we use the xor dataset:

.. code-block:: python

    train_x, train_y, eval_x, eval_y = gt.dataset_generator.xor_dataset()


The Tsetlin Machine is the core of Green Tsetlin. Here a dense (regular) TM:

.. code-block:: python

    tm = gt.TsetlinMachine(n_literals=train_x.shape[1],
                       n_clauses=5,
                       n_classes=2,
                       s=3.0,
                       threshold=42,
                       literal_budget=4)


And here is the sparse TM.

.. code-block:: python

    tm = gt.SparseTsetlinMachine(n_literals=train_x.shape[1],
                             n_clauses=5,
                             n_classes=2,
                             s=3.0,
                             threshold=42,
                             literal_budget=4)

With the green\_tsetlin Trainer we can wrap the TM and train it.

.. code-block:: python
    
    trainer = gt.Trainer(tm, seed=42, n_jobs=1)
    trainer.set_train_data(train_x, train_y)
    trainer.set_eval_data(eval_x, eval_y)
    
    results = trainer.train() # can also be accessed by trainer.results



Exporting and importing models
--------------------------------------------------

After training, the Tsetlin Machine can easily be exported:

.. code-block:: python

    tm.save_state("tsetlin_state.npz")

And imported to continue training or use for inference:

.. code-block:: python

    tm_trained = gt.TsetlinMachine.load_state("tsetlin_state.npz")

    predictor = tm_trained.get_predictor()
    predictor.predict([0, 1, 1, 1])

inference
------------

Using the trained TM for inference lets us predict and explain the prediction. 
This means, given a set of features, we can see which features 
was important for that specific prediction.

Literal explanations
~~~~~~~~~~~~~~~~~~~~~

First we have to get the predictor class. We can get explanations on literals, features or both. If a feature is divided into more than one literal,
it would be insightful to instead use feature explanation. For this example, one feature is one literal, so we use literals.

.. code-block:: python
    
    tm_trained = gt.TsetlinMachine.load_state("tsetlin_state.npz")
    predictor = tm_trained.get_predictor(explanation="literals", exclude_negative_clauses=False)

Then, we want to test on a simple example:

.. code-block:: python
    
    example = [0, 1, 1, 1]
    y_pred, expl = predictor.predict_and_explain(example)

Showing the explanation gives on insight in what features were important.

.. code-block:: python

    for i, (f, w) in enumerate(zip(example, expl)):
        print(f"feature {i}:{f} - {w}")

.. code-block:: none

    feature 0:0 - 124
    feature 1:1 - 192
    feature 2:1 - 0
    feature 3:1 - 0


Feature explanation
~~~~~~~~~~~~~~~~~~~~

Head to the Iris tutorial to see how feature explanation is used.

.. toctree::
   :maxdepth: 2

   iris


Exporting predictor as c program
--------------------------------------------------

With a trained TM we can export the predictor as c program:

.. code-block:: python

    predictor = tm.get_predictor(explanation="literals", exclude_negative_clauses=False)
    predictor.export_as_program("xor_tm_dense.h")



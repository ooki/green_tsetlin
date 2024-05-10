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
    
    trainer.train()

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

Something explainstuff
How to : Inference
=================== 

With a trained Tsetlin Machine we can utilize the interpretable capabilities of the Tsetlin Machine.

.. code-block:: python

    import green_tsetlin as gt

    tm = gt.TsetlinMachine.load_state("tsetlin_state.npz")
    predictor = tm.get_predictor(explanation="features")

    y_pred, expl = predictor.predict_and_explain([0, 1, 1, 1])

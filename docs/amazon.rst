Amazon : large scale multi-class text sentiment classification 
===============================================================


Here we show how green\_tsetlin Sparse Tsetlin Machine can be leveraged for training on the **Amazon Review sentiment dataset** (https://jmcauley.ucsd.edu/data/amazon/).  


Extract the desired number of documents, and process for Sparse Tsetlin Machine training.

.. code-block:: python

    def amazon_iterator(data_path, num_documents):
        reviews = []
        labels = []
        with gzip.open(data_path, mode="rt") as zp:
            for i, line in enumerate(zp):
                if i >= num_documents:
                    break
                try:
                    d = json.loads(line)
                    reviews.append(d['reviewText'])
                    labels.append(int(d['overall']))
                except (json.decoder.JSONDecodeError, KeyError):
                    continue

        return reviews, np.array(labels, dtype=np.uint32) - 1
        

    vectorizer = CountVectorizer(
        analyzer = 'word',
        binary=True,
        ngram_range=(1, 3),
        max_features=None,
        max_df=0.80,
        min_df=3,
        dtype=np.uint8)

                                 # Make sure that the appropriate datafile is installed
    reviews, Y = amazon_iterator('../All_Amazon_Review.json.gz', num_documents=2_000_000)
    X = vectorizer.fit_transform(reviews)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    SKB = SelectKBest(score_func=chi2, k=1_000_000)
    SKB.fit(x_train, y_train)
 
    x_train = SKB.transform(x_train)
    x_test = SKB.transform(x_test)


Define Sparse Tsetlin Machine structure
 
.. code-block:: python

    stm = gt.SparseTsetlinMachine(n_literals=x_train.shape[1], 
                                  n_clauses=2000, 
                                  n_classes=5, 
                                  s=2.0, 
                                  threshold=5000, 
                                  literal_budget=None, 
                                  boost_true_positives=True, 
                                  dynamic_AL=True)        

    
    stm.active_literals_size = 130
    stm.clause_size = 140
    stm.lower_ta_threshold = -90


Wrap model in green\_tsetlin trainer and train.

.. code-block:: python

    trainer = gt.Trainer(stm, seed=42, n_epochs=10, n_jobs=1, progress_bar=True, feedback_type='uniform')
    trainer.set_train_data(x_train, y_train)
    trainer.set_eval_data(x_test, y_test)

    results = trainer.train()


Results from the training can be exstracted form the trainer object.


.. code-block:: python

    {'train_time_of_epochs': [1722.16, 1685.90, 1664.17], 'best_test_score': 0.639, 
    'best_test_epoch': 1, 'n_epochs': 3, 'train_log': [0.631, 0.632, 0.633], 
    'test_log': [0.638, 0.639, 0.637], 'did_early_exit': False}







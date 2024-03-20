
from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState
from sklearn.utils import shuffle


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

import logging
_LOGGER = logging.getLogger(__name__)


def xor_dataset(noise: float = 0.15, n_train: int = 200, n_test: int = 100, n_literals: int = 4,
                seed: int = 42) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Generate a dataset where y = x[0] XOR x[1], with the rest of the features being dummy variables.
    :param noise:   the probability for a noisy training example
    :param n_train: number of training examples
    :param n_test: number of test examples
    :param n_literals: the number of literals, must be at least 2.
    :param seed: random seed. default: None
    :return: train_x (uint8), train_y (uint32), test_x (uint8), test_y (uint32)
    """
    prng = RandomState(seed)

    if n_literals < 2:
        raise ValueError(F"Cannot create xor dataset with less than 2 literals ({n_literals} specified)")

    train_x = prng.randint(low=0, high=2, size=(n_train, n_literals))
    train_y = np.logical_xor(train_x[:, 0], train_x[:, 1])

    if noise is not None and noise > 0.0:
        flips = prng.random(n_train) < noise
        train_y[flips] = np.logical_not(train_y[flips])

    test_x = prng.randint(low=0, high=2, size=(n_test, n_literals))
    test_y = np.logical_xor(test_x[:, 0], test_x[:, 1])

    return train_x.astype(np.uint8), train_y.astype(np.uint32), test_x.astype(np.uint8), test_y.astype(np.uint32)



def imdb_dataset(imdb_num_words: int = 5000, imdb_index_from: int = 2, features: int = 5000, max_ngram: int = 2, 
                 seed: int = 42, train_size: int = None, test_size: int = None, SKB: bool = True):
    """
    Gets and processed the IMDb dataset used for NLP classification
    """
    
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import keras

    _LOGGER.info("Preparing dataset")
    train, test = keras.datasets.imdb.load_data(num_words=imdb_num_words, index_from=imdb_index_from, seed=seed)
    train_x, train_y = train
    test_x, test_y = test

    if train_size:
        train_x, train_y = shuffle(train_x, train_y, n_samples=train_size, random_state=seed)
    if test_size:
        test_x, test_y = shuffle(test_x, test_y, n_samples=test_size, random_state=seed)


    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    _LOGGER.info("Preparing dataset.... Done!")

    _LOGGER.info("Producing bit representation...")

    id_to_word = {value: key for key, value in word_to_id.items()}

    training_documents = []
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id].lower())

        training_documents.append(terms)

    testing_documents = []
    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id].lower())

        testing_documents.append(terms)

    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, max_ngram),
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents).astype(np.uint8)
    y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents).astype(np.uint8)
    y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    if SKB:

        _LOGGER.info("Selecting Features....")
        SKB = SelectKBest(chi2, k=features)
        SKB.fit(X_train, y_train)

        selected_features = SKB.get_support(indices=True)
        x_train = SKB.transform(X_train).toarray()
        x_test = SKB.transform(X_test).toarray()

        _LOGGER.info("Selecting Features.... Done!")


    return x_train, y_train, x_test, y_test


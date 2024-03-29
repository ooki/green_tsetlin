import random
import pickle
import os 
import uuid

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

    
if __name__ == "__main__":
    X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)
    
    x, y = shuffle(X, y, random_state=42)  
    x = np.where(x.reshape((x.shape[0], 28 * 28)) > 75, 1, 0)
    x = x.astype(np.uint8)
    y = y.astype(np.uint32)
    
    n_examples = x.shape[0]
    n_literals = x.shape[1]
    x.astype(np.uint8).tofile("mnist_x_{}_{}.test_bin".format(n_examples, n_literals))
    y.astype(np.uint32).tofile("mnist_y_{}_{}.test_bin".format(n_examples, n_literals))
    
    print("<done>")
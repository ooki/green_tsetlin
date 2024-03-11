import os 
import pickle

import numpy as np




if __name__ == "__main__":
    
    in_file = "./performance_tests/mnist_dense_results.pkl"
    with open(in_file, "rb") as fp:
        data = pickle.load(fp)
        
        
    for d in data:
        print(d)
        print("-"*50)
        


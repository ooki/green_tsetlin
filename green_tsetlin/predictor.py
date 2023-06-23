
from pathlib import Path
import pickle
import os


import numpy as np




class Predictor:
    def __init__(self):
        pass

    def rules_from_state(self, state_or_path):

        if isinstance(state_or_path, str):
            with open(state_or_path, "rb") as fp:
                state = pickle.load(fp)
        
        else:
            state = state_or_path

        
        

        







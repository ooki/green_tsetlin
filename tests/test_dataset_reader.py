import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path




class TsetlinTransformer:
    def __init__(self, pandas_df : Optional[pd.DataFrame] = None):
        
        self.column_names : list = None
        if pandas_df is not None:
            self.column_names = list(pandas_df.columns)
            
            print(pandas_df.dtypes)
            
            pandas_df.infer_objects()
            
            print(pandas_df.dtypes)
            
        
            
            
            
            
        

# def test_read_dataset():
#     abalone_file = (Path(__file__).parent).parent / "test_data" / "abalone" / "abalone.data"    
    
#     df = pd.read_csv(abalone_file)    
    
    
#     #print(df.head())
    
#     tt = TsetlinTransformer(df)
#     #tt.set_column_names(["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"])
#     # tt.set_column_names(df) # if available from pandas import directly    
#     #tt.set_target("rings")
    
    
if __name__ == "__main__":
    test_read_dataset()    
    print("<done tests:", __file__, ">")
    
    
    
    
    
        
    
    
    
    




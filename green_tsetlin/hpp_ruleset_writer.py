
import pathlib
from green_tsetlin.ruleset import RuleSet




class HppWriter:
    def __init__(self, rs: RuleSet, cpp_compability=True):
        self.rs = rs 
        self.cpp_compability = cpp_compability
        
        
    def to_file(self, path_to_file:str):        
        pass
    
    
    
    def _write_header(self, fp):
        
        
        
        

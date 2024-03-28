
import numpy as np
import green_tsetlin as gt

if __name__ == "__main__":    
    ds = gt.DenseState.load_from_file("./generator_tests/mnist_state.npz")
    rs = gt.RuleSet(False)
    rs.compile_from_dense_state(ds)
    
    for r in rs.rules:
        print(r)
        break
    

TODO:
====

- Fix before release:
    - Rename literal_budget to max_rule_length/size
    - move s_inv and s_min1_inv to state for sparse and dense tm
    - Add SparseForDenseTMInputBlock that lets a dense model run on a sparse model with instanciating the dense version only in prepare_example()
    - ~~load last state if load_best_state is false~~ DONE

- Inference
    - Rewrite Inference:predict() to user a sorted list of on literals (like sparse). 
    - Consider writing a "sparse" backend, that uses a map<> to track literal/feature importance
        as to avoid reseting a large array each prediction.
        Could be faster than a dense version in most cases.
    - Also consider writing a "bit encoded" backend for fast inference speeds.

- Sparse TM:
    - vectorized count votes (as dense)
    - ruleset/predictor 
    - setup hpsearch.py so it works for sparseTM
    - convert sparse to dense in SparseTM
    - fix clang bug on mac
    

- Dense TM:
    - vectorized count votes
    - convolutional?
    - convert dense to sparse in DenseTM

- examples
    - implement a non trainer example
    - do a california housing example

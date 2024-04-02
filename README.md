TODO:
====

- Fix before release:
    - Rename literal_budget to max_rule_length/size
    - move s_inv and s_min1_inv to state for sparse and dense tm
    - save last state if save_best_state is false

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
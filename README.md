<p align="center">
  <img src="docs/image/GT_FRONTPAGE_RTD.png" width="400" height="400">
</p>

Green Tsetlin
==============
### **Installation**
Green Tsetlin can be installed by the following:
```bash
pip install green-tsetlin
```

### **Documentation**
The documentation can be found on: https://green-tsetlin.readthedocs.io/  
Below is some short examples of some of the capabilities of green-tsetlin.


### **Tsetlin Machine**
The Tsetlin Machine is the core of Green Tsetlin.
Here a dense (regular) TM:
```python
import green_tsetlin as gt

tm = gt.TsetlinMachine(n_literals=4,
                       n_clauses=5,
                       n_classes=2,
                       s=3.0,
                       threshold=42,
                       literal_budget=4
                       )
```


### **Sparse Tsetlin Machine**
The Tsetlin Machine is the core of Green Tsetlin.
Here a sparse TM:
```python
import green_tsetlin as gt

tm = gt.SparseTsetlinMachine(n_literals=4,
                             n_clauses=5,
                             n_classes=2,
                             s=3.0,
                             threshold=42,
                             literal_budget=4,
                             )
```


### **Trainer**
Use the Trainer to fit a Tsetlin Machine to the data.  
The trainer class is used for both sparse and dense data and TMs.
```python
import green_tsetlin as gt
        

# set n_jobs = 1 to run single threaded
trainer = gt.Trainer(tm, seed=42, n_jobs=1)

trainer.set_train_data(train_x, train_y)
trainer.set_eval_data(eval_x, eval_y)

trainer.train()
```

### **Exporting Tsetlin Machines**
Exporting trained Tsetlin Machines.
```python
.
.
tm.save_state("tsetlin_state.npz")
```


### **Loading exported Tsetlin Machines**
Loading trained Tsetlin Machines to continue training or use for inference.
```python
.
.
tm.load_state("tsetlin_state.npz")
```

### **Inference**
Inference with trained Tsetlin Machines.
```python
.
.
predictor = tm.get_predictor()
predictor.predict(x)
```

### **Green Tsetlin hpsearch**
With the built-in hyperparameter search you can optimize your Tsetlin Machine parameters.
```python
from green_tsetlin.hpsearch import HyperparameterSearch

hyperparam_search = HyperparameterSearch(s_space=(2.0, 20.0),
                                        clause_space=(5, 10),
                                        threshold_space=(3, 20),
                                        max_epoch_per_trial=20,
                                        literal_budget=(1, train_x.shape[1]),
                                        search_or_use_boost_true_positives=(False, True),
                                        seed=42,
                                        n_jobs=5,
                                        k_folds=4,
                                        minimize_literal_budget=False)

hyperparam_search.set_train_data(train_x, train_y)
hyperparam_search.set_eval_data(test_x, test_y)

hyperparam_search.optimize(n_trials=10)
```





# try:
#     import green_tsetlin_core as gtc
# except ImportError:
#     raise ImportError("Could not import the c++ core library  green_tsetlin_core : please check your binary distribution.")



from green_tsetlin.tsetlin_machine import TsetlinMachine, SparseTsetlinMachine, DenseState, SparseState, allocate_clause_blocks
from green_tsetlin.trainer import Trainer
from green_tsetlin.predictor import Predictor

from green_tsetlin.ruleset import RuleSet
from green_tsetlin.ruleset_writer import Writer

import green_tsetlin.dataset_generator as dataset_generator




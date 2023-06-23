
try:
    import green_tsetlin_core as gtc
except ImportError:
    raise ImportError("Could not import the c++ core library  green_tsetlin_core : please check your binary distribution.")



from green_tsetlin.tsetlin_machine import TsetlinMachine
from green_tsetlin.trainer import Trainer
from green_tsetlin.rules import RulePredictor

import green_tsetlin.dataset_generator as dataset_generator




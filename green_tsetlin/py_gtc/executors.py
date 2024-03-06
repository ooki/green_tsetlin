import numpy as np
from sklearn.utils import shuffle


class SingleThreadExecutor:
    def __init__(self, ib, cbs, feedback_block, n_threads, seed):
        self.m_input_block = ib
        self.m_clause_blocks = cbs
        self.m_feedback_block = feedback_block
        self.m_num_threads = n_threads

        self.m_trainable_clause_blocks = [] 

        for cb in self.m_clause_blocks:

            if(not cb.is_init()):
                raise RuntimeError("All ClauseBlocks must be init() before constructing an Executor()")
            if(cb.get_input_block() is None):
                raise RuntimeError("All ClauseBlocks must be have a InputBlock before constructing an Executor()")
            if(cb.get_feedback() is None):
                raise RuntimeError("All ClauseBlocks must be have a FeedbackBlock before constructing an Executor()")

            if(cb.is_trainable()):
                self.m_trainable_clause_blocks.append(cb) # unsure




    def train_epoch(self):
        self.m_feedback_block.reset_train_predict_counter()
        n_examples = self.get_number_of_examples_ready()

        # train.slice() is next
        self.train_slice(0, n_examples)

        return self.m_feedback_block.get_train_accuracy()

    def train_slice(self, start_index, end_index):
        n_examples = self.get_number_of_examples_ready()

        if (start_index == 0):
            m_index_set = np.arange(n_examples)

            # remember seed
            np.random.shuffle(m_index_set)

        for i in range(end_index):
            self.m_feedback_block.reset()

            example_index = m_index_set[i]

            self.m_input_block.prepare_example(example_index)

            for cb in self.m_clause_blocks:

                if (0): # enable multithread
                    pass

                else:   
                    cb.train_example()

            if (0): # enable multithread
                pass

            # NEXT HERE, do training!
            self.m_feedback_block.process(self.m_input_block.pull_current_label())
            positive_class = self.m_feedback_block.get_positive_class()
            negative_class = self.m_feedback_block.get_negative_class()
            pup = self.m_feedback_block.get_positive_update_probability()                  
            nup = self.m_feedback_block.get_negative_update_probability()


            # Do all the rest of cb prep first!

            for cb in self.m_trainable_clause_blocks:

                if(not cb.is_trainable()): # always false?
                    continue

                if(0): # enable multithread
                    pass

                else:
                    cb.train_update(positive_class, pup, negative_class, nup)

    def eval_predict(self):
        n_examples = self.get_number_of_examples_ready()
        
        outputs = np.zeros(n_examples, dtype=np.int32)

        for i in range(n_examples):

            self.m_feedback_block.reset()
            self.m_input_block.prepare_example(i)

            for cb in self.m_clause_blocks:
                cb.eval_example()

            outputs[i] = self.m_feedback_block.predict()

        return outputs
    
    def eval_predict_multi(self):
        pass

        
    def get_number_of_examples_ready(self):
        return self.m_input_block.get_number_of_examples()


    
class MultiThreadExecutor:
    def __init__(self, ib, cbs, feedback_block, n_threads, seed):
        pass
    
    def train_epoch(self):
        pass
    
    def eval_predict(self):
        pass
    
    def eval_predict_multi(self):
        pass
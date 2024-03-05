import numpy as np
from sklearn.utils import shuffle


class SingleThreadExecutor:
    def __init__(self, ib, cbs, feedback_block, n_threads, seed):
        self.m_input_block = ib
        self.m_clause_blocks = cbs
        self.m_feedback_block = feedback_block
        self.m_num_threads = n_threads
    

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


    def eval_predict(self):
        n_examples = self.get_number_of_examples_ready()
        
        outputs = np.zeros(n_examples, dtype=np.int32)

        for i in range(n_examples):

            self.m_feedback_block.reset()
            self.m_input_block.prepare_example(i)

            for cb in self.m_clause_blocks:
                cb.eval_example()


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
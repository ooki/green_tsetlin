



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
    
    def eval_predict(self):
        pass
    
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
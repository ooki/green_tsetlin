import numpy as np

class FeedbackBlock:
    def __init__(self, n_classes, threshold, seed):
        

        self.m_update_probability = np.zeros(n_classes, dtype=np.float32);     
        self.m_votes = np.zeros(n_classes, dtype=np.int32)


    def reset_train_predict_counter(self):
        self.m_correct_train_predict = 0.0
        self.m_total_train_predict = 0.0
    

    def reset(self):
        self.m_votes.fill(0)


    def register_votes(self, class_votes):
        self.m_votes = class_votes
    

    def process(self):
        pass


class FeedbackBlockMultiLabel:
    def __init__(self, n_classes, threshold, seed):
        pass


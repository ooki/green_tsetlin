import numpy as np

class FeedbackBlock:
    def __init__(self, n_classes, threshold, seed):
        

        self.m_n_classes = n_classes
        self.threshold = threshold

        self.m_update_probability = np.zeros(n_classes, dtype=np.float32);     
        self.m_votes = np.zeros(n_classes, dtype=np.int32)


    def reset_train_predict_counter(self):
        self.m_correct_train_predict = 0.0
        self.m_total_train_predict = 0.0
    

    def reset(self):
        self.m_votes.fill(0)


    def register_votes(self, class_votes):
        self.m_votes = class_votes
    

    def process(self, lable):

        positive_class = lable
        most_votes = self.m_votes[positive_class]
        predicted_class = 0

        for class_k in range(self.m_n_classes):
            votes = self.m_votes[class_k]
            
            if(votes > most_votes):
                most_votes = votes
                predicted_class = class_k

            update_p = np.clip(np.array(votes), -self.threshold, self.threshold)
            update_p = (self.threshold + update_p) / (2 * self.threshold)
            self.m_update_probability[class_k] = update_p

        self.m_total_train_predict += 1.0
        if(predicted_class == positive_class):
            self.m_correct_train_predict += 1.0

        self.m_update_probability[positive_class] = 0.0

        # neg class FNS
        negative_class = np.random.choice(self.m_n_classes, p=self.m_update_probability / self.m_update_probability.sum())
        self.m_update_prob_negative =  self.m_update_probability[negative_class]
        # pos class
        update_p = np.clip(np.array(self.m_votes[positive_class]), -self.threshold, self.threshold)
        update_p = (self.threshold + update_p) / (2 * self.threshold)
        self.m_update_prob_positive = update_p


    def get_train_accuracy(self):
        if(self.m_total_train_predict > 0.0):
            return self.m_correct_train_predict / self.m_total_train_predict
        
        return 0.0

class FeedbackBlockMultiLabel:
    def __init__(self, n_classes, threshold, seed):
        pass


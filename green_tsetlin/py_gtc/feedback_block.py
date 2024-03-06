import numpy as np

class FeedbackBlock:
    def __init__(self, n_classes, threshold, seed):
        
        self.rng = np.random.default_rng(seed)

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

        self.positive_class = lable

        # most_votes = self.m_votes[lable]
        most_votes = -self.threshold
        predicted_class = 0

        for class_k in range(self.m_n_classes):
            votes = self.m_votes[class_k]
            
            if(votes > most_votes):
                most_votes = votes
                predicted_class = class_k

            update_p = np.clip(np.array(votes), -self.threshold, self.threshold)
            update_p = (self.threshold + update_p) / (2 * self.threshold) + 1e-16
            self.m_update_probability[class_k] = update_p

        self.m_total_train_predict += 1.0
        if(predicted_class == self.positive_class):
            self.m_correct_train_predict += 1.0

        self.m_update_probability[self.positive_class] = 0.0

        # neg class FNS
        self.negative_class = self.rng.choice(self.m_n_classes, p=self.m_update_probability / self.m_update_probability.sum())
        self.m_update_prob_negative =  self.m_update_probability[self.negative_class]

        # pos class
        update_p = np.clip(np.array(self.m_votes[self.positive_class]), -self.threshold, self.threshold)
        update_p = (self.threshold - update_p) / (2 * self.threshold) + 1e-16
        self.m_update_prob_positive = update_p


    def predict(self):
        best_k = 0
        best_v = self.m_votes[0]

        # should start 
        for i in range(self.m_n_classes):
            if(self.m_votes[i] > best_v):
                best_k = i
                best_v = self.m_votes[i]

        return best_k


    def get_train_accuracy(self):
        if(self.m_total_train_predict > 0.0):
            return self.m_correct_train_predict / self.m_total_train_predict
        
        return 0.0
    

    def get_positive_class(self):
        return self.positive_class

    def get_negative_class(self):
        return self.negative_class

    def get_positive_update_probability(self):
        return self.m_update_prob_positive

    def get_negative_update_probability(self):
        return self.m_update_prob_negative


class FeedbackBlockMultiLabel:
    def __init__(self, n_classes, threshold, seed):
        pass




import numpy as np


class DenseInputBlock:
    def __init__(self, n_literals):
        self.n_literals = n_literals

        self.m_num_labels_per_example = 0

        self.m_num_examples = 0

        self.m_labels = None
        self.m_data = None 


    def get_number_of_examples(self):
        return self.m_num_examples
    

    def set_data(self, x: np.array, y: np.array):
        
        self.m_num_examples = x.shape[0]

        if(x.shape[1]!=self.n_literals):
            raise RuntimeError("Number of literals does not match the data provided in set_data().")

        # not sure if this is correct    
        self.m_data = x

        if(y.shape[0] == 0):
            self.m_labels = None

        else:
            if(y.shape[0] != self.m_num_examples):
                raise RuntimeError("Number of examples in labeles does not match number of examples provided in set_data().")

            if(y.ndim == 1):
                self.m_num_labels_per_example = 1
            else:
                self.m_num_labels_per_example = y.shape[1]

        # not sure if this is correct    
        self.m_labels = y


    def prepare_example(self, index):
        
        if(self.m_labels is not None):
            self.m_current_label = self.m_labels[index*self.m_num_labels_per_example] # unsure about this, stems to id.set_data() m_lable and m_data

        self.m_current_example = self.m_data[index*self.m_num_labels_per_example] # unsure about this, stems to id.set_data() m_lable and m_data

    def pull_current_example(self):
        return self.m_current_example
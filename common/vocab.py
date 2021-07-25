import numpy as np

class Vocab:
    def __init__(self):
        self.vocab_size = None


    def build_vocab(self, data_dict):
        '''
        data_dict["train"]: n_timestamps x n_dims 
        '''
        index = np.unique(data_dict["train_tokens"])
        self.vocab_size = index + 1
        print("# of Discretized tokens: {}".format(self.vocab_size))

        return self.vocab_size
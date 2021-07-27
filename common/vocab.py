import string
import numpy as np
from IPython import embed


class Vocab:
    def __init__(self):
        self.vocab_size = None

    def build_vocab(self, data_dict):
        """
        data_dict["train"]: n_timestamps x n_dims
        """
        vocabs = np.unique(data_dict["train_tokens"])
        self.vocab_size = len(vocabs) + 1  # +1 means oov
        print("Vocab size: {}".format(self.vocab_size))

        self.word2index = {word: idx for idx, word in enumerate(vocabs, 1)}

        self.label2idx = {}
        char2idx = {ch: idx for idx, ch in enumerate(string.ascii_letters)}
        for k, v in self.word2index.items():
            self.label2idx[v] = char2idx[k.split("_")[0]]
        return self.vocab_size

    def transform(self, data_dict):
        print("Transforming tokens to numbers")

        def convert_matrix(arr):
            tmp_arr = []
            for row in arr:
                tmp_arr.append(list(map(lambda x: self.word2index.get(x, 1), row)))
            return np.array(tmp_arr)

        data_dict["train"] = convert_matrix(data_dict["train_tokens"])
        data_dict["test"] = convert_matrix(data_dict["test_tokens"])
        return data_dict
import numpy as np


class DataSet():
    # the length of given data must be even
    def __init__(self, data, data_size, batch_size):
        self.data_ = data
        self.size_ = data_size
        self.bs_ = batch_size
        self.hs_ = int(batch_size / 2)

    def random_batch(self):
        indexes = np.random.randint(
            low=0, high=int(self.size_ / 2), size=(self.hs_)
        )
        right = []
        for i in range(len(indexes)):
            indexes[i] *= 2
            right.append(indexes[i] + 1)
        right = np.asarray(right, dtype=np.int)
        indexes = np.concatenate((indexes, right), axis=0)
        res = {}
        for k in self.data_:
            res[k] = self.data_[k][indexes]
        
        return res, indexes

    def adjust_e_vector(self, new_e, indexes):
        for i in range(len(new_e)):
            idx = indexes[i]
            self.data_['e_vector'][idx] = new_e[i]


if __name__ == '__main__':
    data_set = DataSet(np.random.rand(1000, 5), 16)
    _, indexes = data_set.random_batch()
    length = int(len(indexes) / 2)
    print(indexes[:length])
    print(indexes[length:])

import random
import config
import numpy


# each object has data stored in array as ...key, val, key, val, ...
class NumKeyValData:
    def __init__(self, seed, n_keys, ratio):
        self.access_count = 0
        self.n_keys = n_keys
        self.ratio = ratio
        self.bucket_array = []
        self.data_array = []
        self.seed = seed
        self.size = 0
        self._create_bucket_array()
        self._create_data_array()

    def read(self, index):
        self.access_count += 1
        if len(self.data_array) < index * 2 + 1:
            raise IndexError("Not enough data present")
        return self.data_array[index * 2]

    def _create_bucket_array(self):
        random.seed(self.seed)
        for _ in range(0, self.n_keys):
            if random.uniform(0, 1) < self.ratio:
                self.bucket_array.append(random.randrange(
                    config.MAX_BLOCK_SIZE) + 1)
            else:
                self.bucket_array.append(1)

    def _create_data_array(self):
        for i in range(len(self.bucket_array)):
                for j in range(self.bucket_array[i]):
                    self.data_array.append(i)
                    self.data_array.append(0)
        self.size = int(len(self.data_array)/2)

    def print_truncated(self):
        print("{} ... (avg: {})\n".format(self.bucket_array[0:30],
                                    numpy.mean(self.bucket_array)))

    def get_access_count(self):
        return self.access_count

    def reset_access_count(self):
        self.access_count = 0

import random
import config
import numpy


def generate():
    all_data = []
    for run in range (config.N_RUNS):
        run_data = []
        seed = run
        for interpolation in range(config.N_INTERPOLATIONS):
            size = config.N_KEYS
            ratio = 1/config.N_INTERPOLATIONS * run
            interpolation_data = NumKeyValData(seed, size, ratio)
            run_data.append(interpolation_data)
        all_data.append(run_data)


# each object has data stored in array as ...key, val, key, val, ...
class NumKeyValData:
    def __init__(self, seed, size, ratio):
        self.access_count = 0
        self.size = size
        self.ratio = ratio
        self.bucket_array = []
        self.data_array = []
        self.seed = seed
        self._create_bucket_array()
        self._create_data_array()

    def read(self, index):
        self.access_count += 1
        if len(self.data_array < index * 2 + 1):
            raise IndexError("Not enough data present")
        return self.data_array[index * 2]

    def _create_bucket_array(self):
        random.seed(self.seed)
        for _ in range(0, self.size):
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

    def print_truncated(self):
        for x in range(len(self.data_array)):
            print("block {}: {} ... (avg: {})\n"
                  .format(x, self.data_array[x][0:30],
                          numpy.mean(self.data_array[x])))

import random
import config
import numpy
import csv
from scipy import stats


# each object has data stored in array as ...key, val, key, val, ...
class NumKeyValData:
    def __init__(self):
        self.access_count = 0
        self.n_keys = 0
        self.ratio = 0.0
        self.bucket_array = []
        self.data_array = []
        self.seed = 0
        self.size = 0

        # For linear regression:
        self.slope = 0.0
        self.intercept = 0.0

    def create(self, seed, n_keys, ratio):
        self.n_keys = n_keys
        self.ratio = ratio
        self.seed = seed

        self.bucket_array = []
        self.data_array = []
        self._create_bucket_array()
        self._create_data_array()

        self._linear_regression()

    def load(self, filename):
        with open(filename, 'r', newline='') as data_file:
            csv_reader = csv.reader(data_file, quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                self.data_array.append(int(row[0]))
            self.size = len(self.data_array)

    def save(self, filename):
        with open(filename, 'w', newline='') as data_file:
            wr = csv.writer(data_file, quoting=csv.QUOTE_NONE)
            for i in range(self.size):
                wr.writerow([self.data_array[i], i])

    def read(self, index):
        self.access_count += 1
        if len(self.data_array) < index + 1:
            raise IndexError("Not enough data present")
        return self.data_array[index]

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
        self.size = len(self.data_array)

    def _linear_regression(self):
        self.slope, self.intercept, _, _, _ = stats.linregress(
            self.data_array, range(len(self.data_array)))

    def print_truncated(self):
        print("{} ... (avg: {})\n".format(self.data_array[0:30],
                                    numpy.mean(self.data_array)))

    def get_access_count(self):
        return self.access_count

    def reset_access_count(self):
        self.access_count = 0

# --------------------------------------------------------------------
# datatypes.py - classes for synthetic evaluation data.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import random
import config
import numpy
import csv
from scipy import stats


# each object has data stored in array as ...key, val, key, val, ...
class NumKeyValData:
    """
    Class for generating and using representative datasets for testing.

    """
    def __init__(self):
        """
        Initialises the data and all relevant properties.

        """
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
        """
        Creates synthetic integer dataset.

        :param seed: seed for Random Number Generator
        :param n_keys: number of keys in the dataset
        :param ratio: ratio of multiple indices per keys

        """
        self.n_keys = n_keys
        self.ratio = ratio
        self.seed = seed

        self.bucket_array = []
        self.data_array = []
        self._create_bucket_array()
        self._create_data_array()

        self._linear_regression()

    def load(self, filename):
        """
        Loads a dataset from a csv file.

        :param filename: the path and filename of the csv file.

        """
        with open(filename, 'r', newline='') as data_file:
            csv_reader = csv.reader(data_file, quoting=csv.QUOTE_NONE)
            values = set()
            for row in csv_reader:
                value = int(row[0])
                values.add(value)
                self.data_array.append(value)
            self.size = len(self.data_array)
            self.n_keys = len(values)

    def save(self, filename):
        """
        Saves a dataset as a csv file.

        :param filename: path and filename of the csv file.

        """
        with open(filename, 'w', newline='') as data_file:
            wr = csv.writer(data_file, quoting=csv.QUOTE_NONE)
            for i in range(self.size):
                wr.writerow([self.data_array[i], i])

    def read(self, index):
        """
        returns the record (the key) at the specified index and records this
        action as a data access.

        :param index: the index of the record to be returned

        :return: the key at this index

        """
        self.access_count += 1
        if len(self.data_array) < index + 1:
            raise IndexError("Not enough data present")
        elif index < 0:
            raise IndexError("negative index detected")
        return self.data_array[index]

    def _create_bucket_array(self):
        """
        Creates an array based on which the dataset will be built.
        The size of a bucket (each element of the bucket array) signifies the
        number of indexes corresponding to a key.

        """
        random.seed(self.seed)
        for _ in range(0, self.n_keys):
            if random.uniform(0, 1) < self.ratio:
                self.bucket_array.append(random.randrange(
                    config.MAX_BLOCK_SIZE) + 1)
            else:
                self.bucket_array.append(1)

    def _create_data_array(self):
        """
        Creates the dataset from the bucket array.

        """
        for i in range(len(self.bucket_array)):
                for j in range(self.bucket_array[i]):
                    self.data_array.append(i)
        self.size = len(self.data_array)

    def _linear_regression(self):
        """
        Does a linear regression on the data.

        """
        self.slope, self.intercept, _, _, _ = stats.linregress(
            self.data_array, range(len(self.data_array)))

    def print_truncated(self):
        """
        Primts the first 30 elements of the data array to the console.

        """
        print("{} ... (avg: {})\n".format(self.data_array[0:30],
                                          numpy.mean(self.data_array)))

    def get_access_count(self):
        """
        Returns the number of times the data has been read.

        :return: the access count since the last reset

        """
        return self.access_count

    def reset_access_count(self):
        """
        Resets the access count of the data.

        """
        self.access_count = 0

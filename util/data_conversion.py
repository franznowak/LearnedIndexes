# --------------------------------------------------------------------
# csv_to_training_data.py - Script converting real world data into usable
# training data.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

"""
Script that takes a filename of a csv that has the keys in its
first column or a binary file and returns training data of key, index pairs.

"""
import sys

import pandas as pd
import numpy as np


filename = sys.argv[1]
filetype = sys.argv[2]

if filetype == "csv":
    dataset = pd.read_csv(filename, comment='\t',
                          sep=",", skipinitialspace=True)

    keys = dataset.iloc[:, 0].astype('int64')

    with open(filename + "_training", "w") as file:
        for i in range(len(keys.index)):
            file.write("{},{}\n".format(keys[i], i))

elif filetype == "binary":
    data = np.fromfile(filename, dtype=np.uint32)

    with open(filename + "_training", "w") as file:
        for i in range(len(data)):
            file.write("{},{}\n".format(data[i], i))

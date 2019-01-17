from __future__ import absolute_import, division, print_function

import pathlib

import numpy as np

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    dataset_path = '/Users/franz/.keras/datasets/random.data'

    column_names = ['key', 'index']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    print(dataset.tail())





if __name__ == "__main__":
    main()
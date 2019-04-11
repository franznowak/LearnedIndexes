# --------------------------------------------------------------------
# array_index.py - class for making predictions using array_index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import util.search as searcher


def array_predict(input_data, key):
    """
    
    :param input_data: the dataset
    :param key: the key whose position is to be predicted.

    :return: access count
    """
    prediction = int(input_data.size/input_data.n_keys * key)
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()


def linear_regression(input_data, key):
    """
    Deprecated. uses data's own slope and intercept data to return linear
    regression.

    """
    prediction = int(input_data.slope * key + input_data.intercept)
    prediction = max(0, min(prediction, input_data.size-1))
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()

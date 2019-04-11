import util.search as searcher


def get_search_access_count(input_data, prediction, key):
    """
    Searches for the key and returns the data accesses that were necessary to
    find it.

    :param input_data: the data of type NumKeyValData
    :param prediction: the estimated position of the key in the data
    :param key: the key to be found

    :return: number of accesses used in search

    """
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()

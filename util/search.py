# --------------------------------------------------------------------
# search - various search functions for finding an index by key in data.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from enum import Enum, auto
import config
import numpy as np


class SearchType(Enum):
    """ Enum for all currently supported search types"""
    LINEAR = auto()
    BINARY = auto()
    EXPONENTIAL = auto()


def search(data, start_index, target_value, search_type: SearchType = None):
    """
    Searches the data for an index.

    :param data: the data to be searched
    :param start_index: the initial estimate of the index
    :param target_value: the key that we are searching for
    :param search_type: SearchType, type of search to be used

    """
    if search_type is None:
        search_type = config.SEARCH

    if search_type == SearchType.LINEAR:
        linear_search(data, start_index, target_value)
    elif search_type == SearchType.BINARY:
        binary_search(data, target_value)
    elif search_type == SearchType.EXPONENTIAL:
        exponential_search(data, start_index, target_value)
    else:
        raise Exception("Unknown search type")


def linear_search(data, start_index, target_value):
    """
    Search linearly through the data, takes linear time in size of dataset.

    :param data: the data to be searched
    :param start_index: the initial estimate of the index
    :param target_value: the key that we are searching for

    """
    index = max(0, min(data.size-1, start_index))
    value = data.read(index)
    while value < target_value:
        index += 1
        value = data.read(index)
    while value > target_value:
        index -= 1
        value = data.read(index)
    if value != target_value:
        raise Exception("value " + str(target_value) + " not found!")
    return index


def binary_search(data, target_value, left=0, right=None):
    """
    Searches the whole dataset using binary search.

    :param data: the data to be searched
    :param target_value: the key that we are searching for
    :param left: leftmost index in data
    :param right: rightmost index in data

    :return: index

    """
    if right is None:
        right = data.size - 1
    left = max(0, left)
    right = min(right, data.size - 1)
    while True:
        if right < left:
            raise Exception("value not found!")
        index = int(np.floor((right + left) / 2))
        value = data.read(index)
        if value < target_value:
            left = index + 1
        elif value > target_value:
            right = index - 1
        else:
            return index


def exponential_search(data, start_index, target_value):
    """
    Searches around a predicted record in  exponentially larger areas

    :param data: the data to be searched
    :param start_index: the initial estimate of the index
    :param target_value: the key that we are searching for

    :return: index

    """
    index = max(0, min(data.size-1, start_index))
    value = data.read(index)
    jump = 2
    if value < target_value:
        while value < target_value:
            jump *= 2
            value = data.read(min(index + jump, data.size-1))
        return binary_search(data, target_value, index+jump/2, index+jump)
    elif value > target_value:
        while value > target_value:
            jump *= 2
            value = data.read(max(index - jump, 0))
        return binary_search(data, target_value, index-jump, index-jump/2)
    else:
        return index

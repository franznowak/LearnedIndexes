# --------------------------------------------------------------------
# search.py - various search functions for finding an index by key in data.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

from enum import Enum, auto
import config
import numpy as np

from custom_exceptions import KeyNotFoundError


class SearchType(Enum):
    """ Enum for all currently supported search types"""
    LINEAR = auto()
    BINARY = auto()
    QUATERNARY = auto()
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
    elif search_type == SearchType.QUATERNARY:
        quaternary_search(data, target_value)
    elif search_type == SearchType.EXPONENTIAL:
        exponential_search(data, start_index, target_value)
    else:
        raise KeyNotFoundError("Unknown search type")


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
        raise KeyNotFoundError("value " + str(target_value) + " not found!")
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
            raise KeyNotFoundError("value not found!")
        index = int(np.floor((right + left) / 2))
        value = data.read(index)
        if value < target_value:
            left = index + 1
        elif value > target_value:
            right = index - 1
        else:
            return index


def quaternary_search(data, target_value, left=0, right=None):
    """
    Searches the whole dataset using quatenary search.

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
        if right < 0:
            raise KeyNotFoundError("value not found!")
        q1 = int(left + (right-left)/4)
        q2 = int(left + (right-left)/2)
        q3 = int(left + 3*(right-left)/4)

        v1 = data.read(q1)
        v2 = data.read(q2)
        v3 = data.read(q3)

        if target_value == v1:
            return q1
        elif target_value == v2:
            return q2
        elif target_value == v3:
            return q3
        elif target_value < v1:
            right = q1 - 1
        elif target_value < v2:
            left = q1 + 1
            right = q2 - 1
        elif target_value < v3:
            left = q2 + 1
            right = q3 - 1
        else:
            left = q3 + 1


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
    jump = 1
    if value < target_value:
        while value < target_value:
            jump *= 2
            value = data.read(min(index + jump, data.size-1))
        return binary_search(data, target_value, index+int(jump/2), index+jump)
    elif value > target_value:
        while value > target_value:
            jump *= 2
            value = data.read(max(index - jump, 0))
        return binary_search(data, target_value, index-jump, index-int(jump/2))
    else:
        return index

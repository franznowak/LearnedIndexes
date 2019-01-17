from enum import Enum, auto
import config
import numpy as np


class SearchType(Enum):
    LINEAR = auto()
    BINARY = auto()
    EXPONENTIAL = auto()


def search(data, start_index, target_value, search_type=None):
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
    index = start_index
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
    if right is None:
        right = data.size - 1
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
    index = start_index
    value = data.read(index)
    jump = 2
    if value < target_value:
        while value < target_value:
            jump *= 2
            index += jump
            value = data.read(index)
        return binary_search(data, target_value, index-jump, index)
    elif value > target_value:
        while value > target_value:
            jump *= 2
            index -= jump
            value = data.read(index)
        return binary_search(data, target_value, index, index + jump)
    else:
        return index

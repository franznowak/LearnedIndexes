import enum
from enum import Enum
import config


class SearchType(Enum):
    LINEAR = enum.auto
    BINARY = enum.auto
    EXPONENTIAL = enum.auto


def search(data, start_index, target_value):
    if config.SEARCH == SearchType.LINEAR:
        linear_search(data, start_index, target_value)
    elif config.SEARCH == SearchType.BINARY:
        binary_search(data, start_index, target_value)
    elif config.SEARCH == SearchType.EXPONENTIAL:
        exponential_search(data, start_index, target_value)
    else:
        raise Exception("Unknown search type")


def linear_search(data, start_index, target_value):
    index = start_index
    while data.read(index) < target_value:
        index += 2
    while data.read(index) > target_value:
        index -= 2
    if data.read(index) != target_value:
        raise Exception("key not found!")


def binary_search(data, start_index, target_value):
    # TODO @@@
    pass


def exponential_search(data, start_index, target_value):
    # TODO @@@
    pass

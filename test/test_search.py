# --------------------------------------------------------------------
# test_search.py - test the searcher module.
# December 2018 - May 2019 - Franz Nowak
# --------------------------------------------------------------------

import util.search
import unittest

from custom_exceptions import KeyNotFoundError


class SearchTest(unittest.TestCase):
    """Test various local search algorithms"""
    def test_binary_search(self):
        """finding a key using binary search"""
        data = MockData([1, 2, 3, 4, 5])
        key = data.data[3]
        self.assertEqual(util.search.binary_search(data, key), 3)

    def test_binary_search_negative(self):
        """key not found"""
        data = MockData([1, 2, 3, 4, 5])
        key = 0
        self.assertRaises(KeyNotFoundError,util.search.binary_search(data, key))

    def test_exponential_search(self):
        """finding a key using exponential search"""
        data = MockData([1, 2, 3, 4, 5])
        key = data.data[3]
        self.assertEqual(util.search.exponential_search(data, 2,  key), 3)

    def test_exponential_search_negative(self):
        """key not found"""
        data = MockData([1, 2, 3, 4, 5])
        key = 0
        self.assertRaises(KeyNotFoundError,
                          util.search.exponential_search(data, 2, key))

    def test_quaternary_search(self):
        """finding a key using quaternary search"""
        data = MockData([1, 2, 3, 4, 5])
        key = data.data[3]
        self.assertEqual(util.search.quaternary_search(data, key), 3)

    def test_quaternary_search_negative(self):
        """key not found"""
        data = MockData([1, 2, 3, 4, 5])
        key = 0
        self.assertRaises(KeyNotFoundError,
                          util.search.quaternary_search(data, key))


class MockData:
    """A mock data class for search reads"""
    def __init__(self, data):
        self.data=data

    def read(self, index):
        return self.data[index]

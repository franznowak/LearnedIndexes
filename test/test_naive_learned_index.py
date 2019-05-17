# --------------------------------------------------------------------
# test_naive_learned_index.py - test the naive index.
# December 2018 - May 2019 - Franz Nowak
# --------------------------------------------------------------------
import unittest
from index.naive_learned_index import Model


class NaiveTest(unittest.TestCase):
    """Test Naive Learned Index"""

    def test_model_creation(self):
        """make sure keras model creation works correctly"""
        complexity = [100, 100, 1]
        model = Model._build_model(complexity, 0.001)
        self.assertEqual(len(model.layers), len(complexity) + 1)

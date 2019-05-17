# --------------------------------------------------------------------
# test_hybrid_learned_index.py - test the hybrid index.
# December 2018 - May 2019 - Franz Nowak
# --------------------------------------------------------------------


import unittest

import config
from index.hybrid_learned_index import HybridLearnedIndex


class HybridTest(unittest.TestCase):
    """Test Hybrid Training Algorithm"""
    def test_select_model(self):
        """Normal model selection, should select proportional."""
        self.hybrid_index = HybridLearnedIndex(config.RECURSIVE_SHAPE,
                                                config.RECURSIVE_COMPLEXITY,
                                               1000)
        self.hybrid_index.data_size = 100000
        current_stage = 0
        current_prediction = 10000
        next_index = self.hybrid_index._get_next_model_index(current_stage,
                                                    current_prediction)
        self.assertEqual(next_index, 1)

    def test_select_model_borderline(self):
        """Ensure correct rounding down at border."""
        self.hybrid_index = HybridLearnedIndex(config.RECURSIVE_SHAPE,
                                                config.RECURSIVE_COMPLEXITY,
                                               1000)
        self.hybrid_index.data_size = 100000
        current_stage = 0
        current_prediction = 9999
        next_index = self.hybrid_index._get_next_model_index(current_stage,
                                                    current_prediction)
        self.assertEqual(next_index, 0)

    def test_prediction_negative(self):
        """Negative predictions need to be sanitized, i.e. select first 
        model."""
        self.hybrid_index = HybridLearnedIndex(config.RECURSIVE_SHAPE,
                                                config.RECURSIVE_COMPLEXITY,
                                               1000)
        self.hybrid_index.data_size = 100000
        current_stage = 0
        current_prediction = -10000
        next_index = self.hybrid_index._get_next_model_index(current_stage,
                                                                current_prediction)
        self.assertEqual(next_index, 0)

    def test_prediction_too_high(self):
        """Out of bounds predictions need to be sanitized, i.e. select last 
        model."""
        self.hybrid_index = HybridLearnedIndex(config.RECURSIVE_SHAPE,
                                                config.RECURSIVE_COMPLEXITY,
                                               1000)
        self.hybrid_index.data_size = 100000
        current_stage = 0
        current_prediction = 110000
        next_index = self.hybrid_index._get_next_model_index(current_stage,
                                                                current_prediction)
        self.assertEqual(next_index, 9)

# --------------------------------------------------------------------
# __main__.py - script for evaluation all indexes for time and file accesses.
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

import evaluation

"""
Systematically test all indexes and save the prediction times and accesses
for evaluation.

"""

evaluation.measure_predictions_on_synthetic_integers("recursive_learned_index")

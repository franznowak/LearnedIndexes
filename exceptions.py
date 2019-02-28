# --------------------------------------------------------------------
# exceptions.py - custom exceptions
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

"""Custom exceptions"""


class NoPredictionForIndex(Exception):
    """Exception when there are no predictions found for this index"""
    pass


class NoPredictionsForDataset(Exception):
    """Exception for when there are no predictions found for this data set."""
    pass


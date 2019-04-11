# --------------------------------------------------------------------
# li_exceptions.py - custom exceptions
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------

"""Custom exceptions"""


class ModelNotTrainedException(Exception):
    """Exception for when there is an attempt to use a model before training
    it"""
    pass


class NoPredictionForIndex(Exception):
    """Exception when there are no predictions found for this index"""
    pass


class NoPredictionsForDataset(Exception):
    """Exception for when there are no predictions found for this data set"""
    pass


class NoEvaluationImplemented(Exception):
    """Exception for when there is no evaluation for a certain index type"""
    pass


class PlotTypeNotSupported(Exception):
    """Exception for when the visualiser is called with a plot type that is
    not supported"""
    pass

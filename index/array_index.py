# --------------------------------------------------------------------
# array_index.py - class for making predictions using array_index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------


class ArrayIndex:

    @staticmethod
    def predict(input_data, key):
        """
        Predicts the position of the record based on uniform distribution.

        :param input_data: the dataset
        :param key: the key whose position is to be predicted.

        :return: predicted position

        """
        prediction = int(input_data.size/input_data.n_keys * key)
        return max(0, min(input_data.size-1, prediction))

    @staticmethod
    def linear_regression(input_data, key):
        """
        Deprecated. uses data's own slope and intercept data to return linear
        regression.

        """
        prediction = int(input_data.slope * key + input_data.intercept)
        return max(0, min(input_data.size-1, prediction))

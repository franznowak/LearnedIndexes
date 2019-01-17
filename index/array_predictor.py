import util.search as searcher


def array_predict(input_data, key):
    prediction = int(input_data.size/input_data.n_keys * key)
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()


def linear_regression(input_data, key):
    prediction = int(input_data.slope * key + input_data.intercept)
    prediction = max(0, min(prediction, input_data.size-1))
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()

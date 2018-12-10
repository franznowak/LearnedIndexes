import util.data_access as data
import config
import util.search as searcher


def array_predict(input_data, key):
    prediction = int((len(input_data)/config.N_KEYS) * key/2) * 2
    searcher.search(input_data, prediction, key)


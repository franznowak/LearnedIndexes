import util.search as searcher


def get_accesses(input_data, prediction, key):
    searcher.search(input_data, prediction, key)
    return input_data.get_access_count()

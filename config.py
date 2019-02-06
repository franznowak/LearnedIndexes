from util.search import SearchType

# N_KEYS:int - Number of keys in the index
N_KEYS = 100000
# MAX_BLOCK_SIZE:int - Max size of blocks where same key appears more than once
MAX_BLOCK_SIZE = 9
# N_INTERPOLATIONS:int - For calculating a spread of how likely a jump is
N_INTERPOLATIONS = 2
# N_RUNS:int - Number of iterations with different seeds TODO: @@@ before: 100
N_RUNS = 1
# N_SAMPLES:int - Number of points where access is sampled
N_SAMPLES = 1000
# SEARCH:enum - Determine which search to use
SEARCH = SearchType.LINEAR
# FILE_PATH:str - Path where datasets are stored
FILE_PATH = "data/"
# MODEL_PATH:str - Path where models are stored
MODEL_PATH = "index/models/"

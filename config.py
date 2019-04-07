from util.search import SearchType

# N_KEYS:int - Number of keys in the index
N_KEYS = 100000
# MAX_BLOCK_SIZE:int - Max size of blocks where same key appears more than once
MAX_BLOCK_SIZE = 9
# N_INTERPOLATIONS:int - For calculating a spread of how likely a jump is
N_INTERPOLATIONS = 10
# N_RUNS:int - Number of iterations with different seeds TODO: @@@ before: 100
N_RUNS = 10
# N_SAMPLES:int - Number of points where access is sampled
N_SAMPLES = 1000
# SEARCH:enum - Determine which search to use
SEARCH = SearchType.LINEAR
# INDEX:str - Type of index to be used
INDEX = "naive_learned_index"
# STEP_SIZE:float - the step size used in training for the neural network
STEP_SIZE = 0.001

# DATASET:str - Type of data
DATASET = "Integers_100x10x100k"
# DATASET_PATH:str - Path where dataset is stored
DATASET_PATH = "data/datasets/"+DATASET+"/"
# MODEL_PATH:str - Path where naive_models_0 are stored
MODEL_PATH = "data/indexes/"+INDEX+"/"+DATASET+"/"

PREDICTIONS_PATH = "data/predictions/"

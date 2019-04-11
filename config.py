from util.search import SearchType

# N_KEYS:int - Number of keys in the index
N_KEYS = 100000
# MAX_BLOCK_SIZE:int - Max size of blocks where same key appears more than once
MAX_BLOCK_SIZE = 9
# N_INTERPOLATIONS:int - For calculating a spread of how likely a jump is
N_INTERPOLATIONS = 10
# N_RUNS:int - Number of iterations with different seeds (default: 100)
N_RUNS = 1
# N_SAMPLES:int - Number of points where access is sampled
N_SAMPLES = 1000
# SEARCH:enum - Determine which search to use
SEARCH = SearchType.LINEAR
# INDEX:str - Type of index to be used
INDEX = "naive_learned_index"  # TODO: make variable
# STEP_SIZE:float - the step size used in training for the neural network
STEP_SIZE = 0.001

# INTEGER_DATASET:str - The synthetic dataset of integers
INTEGER_DATASET = "Integers_100x10x100k"
# INTEGER_DATASET_PATH:str - Path where dataset is stored
INTEGER_DATASET_PATH = "data/datasets/"+INTEGER_DATASET+"/"
# MODEL_PATH:str - Path where naive_models_0 are stored
MODEL_PATH = "data/indexes/"+INDEX+"/"+INTEGER_DATASET+"/"
# MODEL_PATH:str - Path where prediction results are stored
PREDICTIONS_PATH = "data/predictions/"+INDEX+"/"+INTEGER_DATASET+"/"

# --------------------------------------------------------------------
# prediction.py - class for making predictions using index
# December 2018 - May 2019 Franz Nowak
# --------------------------------------------------------------------
import config
from index import array_index


def predict(index, data):
    predictions = []
    for run in range(config.N_RUNS):
        print("run # " + str(run+1) + "/" + str(config.N_RUNS))
        predictions.append([])
        for inter in range(config.N_INTERPOLATIONS):
            prediction = []
            step = int(config.N_KEYS / config.N_SAMPLES)
            for i in range(0, config.N_KEYS, step):
                prediction.append(array_index.array_predict(data[run][inter],
                                                            i))
                data[run][inter].reset_access_count()
            predictions[run].append(prediction)

    # write to file: print(predictions)


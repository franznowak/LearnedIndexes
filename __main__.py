import config
import util.data as data
import index.array as array

data = data.generate()

predictions = []
for run in range(config.N_RUNS):
    predictions.append([])
    for inter in range(config.N_INTERPOLATIONS):
        prediction = []
        for i in range(0, config.N_KEYS, int(config.N_KEYS / config.N_SAMPLES)):
            prediction.append(array.array_predict(data[run][inter], i))
        predictions[run][inter].append(prediction)

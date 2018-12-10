import config
import util.datagen as datagen
import index.array as array

predictions = []
data = []

datagen.create_data(data)

for i in range(config.N_RUNS):
    predictions.append([])
    for j in range(len(data[i])):
        predictions[i].append([array.array_predict(data[i][j],x)
                               for x in range(0,config.N_KEYS,
                                              int(
                                                  config.N_KEYS/config.
                                                  N_SAMPLES))])
# TODO: fix

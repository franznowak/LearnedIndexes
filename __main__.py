import config
from util.datatypes import NumKeyValData
import index.array as array

all_data = []


def main():
    print("start creating data...")
    generate_data()
    print("Done.")
    for b in all_data[0]:
        b.print_truncated()

    print("Start predictions...")
    predictions = []
    for run in range(config.N_RUNS):
        print("run # " + str(run+1) + "/" + str(config.N_RUNS))
        predictions.append([])
        for inter in range(config.N_INTERPOLATIONS):
            print("i"+str(inter))
            predictions[run].append([])
            prediction = []
            step = int(config.N_KEYS / config.N_SAMPLES)
            for i in range(0, config.N_KEYS, step):
                prediction.append(array.array_predict(all_data[run][inter], i))
                all_data[run][inter].reset_access_count()
            predictions[run][inter].append(prediction)
            print(prediction)

    print("Done.")
    print(predictions)


def generate_data():
    for run in range(config.N_RUNS):
        print("run # " + str(run+1) + "/" + str(config.N_RUNS))
        run_data = []
        seed = run
        for interpolation in range(config.N_INTERPOLATIONS):
            n_keys = config.N_KEYS
            ratio = 1/config.N_INTERPOLATIONS * interpolation
            interpolation_data = NumKeyValData(seed, n_keys, ratio)
            run_data.append(interpolation_data)
        all_data.append(run_data)


if __name__ == "__main__":
    main()

import csv
import config
from util.datatypes import NumKeyValData


def main():
    #generate_all_data()
    data = load_all_data()
    for a in data:
        for b in a:
            print(b.data_array)


FILE_PATH = "training_data/"


def generate_all_data():
    all_data = []
    for run in range(config.N_RUNS):
        print("creating data for run # " + str(run+1) + "/" + str(
            config.N_RUNS))
        run_data = []
        seed = run
        for interpolation in range(config.N_INTERPOLATIONS):
            n_keys = config.N_KEYS
            ratio = 1/config.N_INTERPOLATIONS * interpolation
            interpolation_data = NumKeyValData()
            interpolation_data.create(seed, n_keys, ratio)
            run_data.append(interpolation_data)

            file_name = "run" + str(run) + "inter" + str(interpolation)
            interpolation_data.save(FILE_PATH + file_name)

        all_data.append(run_data)
    return all_data


def load_all_data():
    all_data = []
    for run in range(config.N_RUNS):
        print("loading data for run # " + str(run+1) + "/" + str(
            config.N_RUNS))
        run_data = []
        for interpolation in range(config.N_INTERPOLATIONS):
            interpolation_data = NumKeyValData()
            file_name = "run" + str(run) + "inter" + str(interpolation)
            interpolation_data.load(FILE_PATH + file_name)
            run_data.append(interpolation_data)

        all_data.append(run_data)
    return all_data


if __name__ == "__main__":
    main()

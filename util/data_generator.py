import csv
import config
from util.datatypes import NumKeyValData


def main():
    all_data = generate_data()
    for run in range(len(all_data)):
        print("writing data for run # " + str(run+1) + "/" + str(
            config.N_RUNS))
        for interpolation in range(len(all_data[run])):
            file_name = "run"+str(run)+"inter"+str(interpolation)
            with open("../data/"+file_name, 'w', newline='') as data_file:
                wr = csv.writer(data_file, quoting=csv.QUOTE_NONE)
                for i in range(all_data[run][interpolation].size):
                    wr.writerow([all_data[run][interpolation].read(i), i])


def generate_data():
    all_data = []
    for run in range(config.N_RUNS):
        print("creating data for run # " + str(run+1) + "/" + str(
            config.N_RUNS))
        run_data = []
        seed = run
        for interpolation in range(config.N_INTERPOLATIONS):
            n_keys = config.N_KEYS
            ratio = 1/config.N_INTERPOLATIONS * interpolation
            interpolation_data = NumKeyValData(seed, n_keys, ratio)
            run_data.append(interpolation_data)
        all_data.append(run_data)
    return all_data


if __name__ == "__main__":
    main()

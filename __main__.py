import config
import util.data_generator as datagen
import index.array_predictor as array


def main():
    print("start creating data...")
    all_data = datagen.generate_data()
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
                prediction.append(array.linear_regression(all_data[run][inter], i))
                all_data[run][inter].reset_access_count()
            predictions[run][inter].append(prediction)
            print(prediction)

    print("Done.")
    print(predictions)


if __name__ == "__main__":
    main()

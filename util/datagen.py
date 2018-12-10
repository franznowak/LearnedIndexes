import random
import config
import numpy


def get_bucket_array(bucket_array, size, ratio):
    for _ in range(0, size):
        if random.uniform(0, 1) < ratio:
            bucket_array.append(random.randrange(config.MAX_BLOCK_SIZE) + 1)
        else:
            bucket_array.append(1)


def create_blocks(blocks):
    for i in range(config.N_RUNS):
        random.seed(i)
        block = []
        for x in range(0, config.N_INTERPOLATIONS):
            bucket_array = []
            block.append(get_bucket_array(bucket_array, config.N_KEYS,
                         1/config.N_INTERPOLATIONS * x))
        blocks.append(block)


def create_data(data, blocks):
    for x in range(config.N_RUNS):
        data.append([])
        for i in range(len(blocks[x])):
            data[x].append([])
            for j in range(len(blocks[x][i])):
                for k in range(blocks[x][i][j]):
                    data[x][i].append(j)
                    data[x][i].append(0)


def print_truncated(array):
    for x in range(len(array)):
        print("block {}: {} ... (avg: {})\n".format(x, array[x][0:30],
                                                    numpy.mean(array[x])))

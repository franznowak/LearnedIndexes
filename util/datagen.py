import random
import config


def get_bucket_array(size, ratio):
    bucket_array=[]
    for _ in range(0, size):
        if random.uniform(0,1)<ratio:
            bucket_array.append(random.randrange(config.MAX_BLOCK_SIZE)+1)
        else:
            bucket_array.append(1)
    return bucket_array

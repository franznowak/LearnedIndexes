
class Data:
    def __init__(self):
        self.count = 0

    def read(self, array, index):
        self.count += 1
        return array[index * 2]  # stored in array as ...key, val, key, val, ...


def train_hybrid(threshold:int, stages, NN_complexity, all_data):
    """
    Hybrid end-to-end training algorithm as described in The Case For
    Learned Index Structures, Kraska et al. p8.
    :param threshold:
        max absolute error of NN before other index is used.
    :param stages:
        list of widths of the regression tree for each level.
    :param NN_complexity:
        width and depth of the NNs for each node in the regression tree.

    :return trained_index

    """
    trained_index = []

    M = len(stages)

    tmp_records = [[]]
    tmp_records[0].append(all_data)

    for i in range(0, M):
        trained_index.append([])
        for j in range(0, stages[i]):
            trained_index[i].append([])
            # TODO: replace empty by new NN trained on tmp_records[i][j]
            if i < M:
                for r in tmp_records[i][j]:
                    p = trained_index[i][j].predict(r.key)/stages[i+1]
                    tmp_records[i+1][p].add(r)

    # if error too high (above threshold) replace with b-tree here

    return trained_index

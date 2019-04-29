import pandas as pd
import config
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from custom_exceptions import PlotTypeNotSupported


def create_graphs(predictions_path, graph_path, kind="scatter", timestamp='new'):
    """
    plots a view of the predictions made by an index for a data set and saves
    it to file.

    :param predictions_path:
        path at which the prediction data is stored
    :param graph_path:
        path at which the graph data shall be stored
    :param kind:
        type of plot to be used: "line", "scatter" or "hist2d"
        (line collapses points to their average)
    :param timestamp:
        timestamp of the predictions, default: 'new' for latest

    """

    # prediction time
    filename = timestamp + "_pred_times"
    pred = pd.read_csv(
        predictions_path + filename + ".csv", header=None)
    plot(pred, kind, title="Prediction time per key in microseconds",
         ylabel="time in microseconds", graph_path=graph_path,
         filename=filename)

    # search time
    filename = timestamp + "_search_times"
    search = pd.read_csv(
        predictions_path + filename + ".csv", header=None)
    plot(search, kind, title="Search time per key in microseconds",
         ylabel="time in microseconds", graph_path=graph_path,
         filename=filename)

    # total time
    filename = timestamp + "_total_times"
    search = pd.read_csv(
        predictions_path + filename + ".csv", header=None)
    plot(search, kind, title="Total index time in microseconds",
         ylabel='time in microseconds', graph_path=graph_path,
         filename=filename)

    # number of reads
    filename = timestamp + "_reads"
    reads = pd.read_csv(
        predictions_path + filename + ".csv", header=None)
    plot(reads, kind, title='Average reads required for search',
         ylabel="number of reads", graph_path=graph_path, filename=filename)


def plot(data, kind, title, ylabel, graph_path, filename, binsize=[50,10]):
    """
    Draws a plot of the data.

    :param data: the data to be plotted (indexed data points)
    :param kind: "scatter" or "hist2d" or "line" (line averages)
    :param title: title of the plot
    :param ylabel: label for the y axis (x is entropy level)
    :param graph_path: path at which the graph data shall be stored
    :param filename: name of the file where graph will be saved
    :param binsize: size of the bins for heat maps

    """
    fig, ax = plt.subplots()
    ax.set(xlabel='entropy', ylabel=ylabel, title=title)
    ax.grid()
    if kind == "scatter":
        plt.scatter(data[0], data[1])
    elif kind == "hist2d":
        plt.hist2d(data[0], data[1], bins=binsize, cmap=cm.Reds, cmin=1)
    elif kind == "line":
        xs, ys, errs = average(data)
        plt.errorbar(xs, ys, yerr=errs)
    else:
        raise(PlotTypeNotSupported())
    plt.savefig(graph_path + filename)
    plt.close()


def average(data):
    """
    Collapses pandas.DataFrame on first column and averages second column.

    :param data: pandas.DataFrame with two columns

    :return: pair of lists, first column, second column

    """
    values = {}
    std = {}
    for i in data.index:
        x = data.at[i, 0]
        y = data.at[i, 1]
        if x in values.keys():
            values[x].append(y)
        else:
            values[x] = [y]
    for x in values.keys():
        std[x] = np.std(values[x])
        values[x] = np.average(values[x])
    return values.keys(), values.values(), std


if __name__ == "__main__":
    predictions_path = "/Users/franz/PycharmProjects/LearnedIndexes/data/pred" \
                       "ictions/binary_search/Integers_100x10x100k/"
    graph_path = "/Users/franz/PycharmProjects/LearnedIndexes/data/graphs" \
                 "/binary_search/Integers_100x10x100k/"
    create_graphs(predictions_path, graph_path, kind="line", timestamp='linear')

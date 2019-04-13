import pandas as pd
import config
import time
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from custom_exceptions import PlotTypeNotSupported


def create_graphs(predictions_path, graph_path, kind="scatter", timestamp='new',
                  showAverage=False):
    """
    plots a view of the predictions made by an index for a data set and saves
    it to file.

    :param predictions_path:
        path at which the prediction data is stored
    :param graph_path:
        path at which the graph data shall be stored
    :param kind:
        type of plot to be used: "scatter" or "hist2d"
    :param timestamp:
        timestamp of the predictions, default: 'new' for latest
    :param showAverage:
        specifies if there should be a line that shows the average of the runs

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


def plot(data, kind, title, ylabel, graph_path, filename, binsize=50):
    """
    Draws a plot of the data.

    :param data: the data to be plotted (indexed data points)
    :param kind: "scatter" or "hist2d"
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
        plt.hist2d(data[0], data[1], bins=binsize, cmap=cm.jet)
    else:
        raise(PlotTypeNotSupported())
    plt.savefig(graph_path + filename)
    plt.close()

import pandas as pd
import config
import time
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from custom_exceptions import PlotTypeNotSupported


def create_graphs(predictions_path, graph_path, kind, timestamp='new',
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
    pred = pd.read_csv(
        predictions_path + timestamp + "_pred_times.csv", header=None)
    plot(pred, kind, title="Prediction time per key in microseconds",
         ylabel="time in microseconds", graph_path=graph_path)

    # search time
    search = pd.read_csv(
        predictions_path + timestamp + "_search_times.csv", header=None)
    plot(search, kind, title="Search time per key in microseconds",
         ylabel="time in microseconds", graph_path=graph_path)

    # search time
    search = pd.read_csv(
        predictions_path + timestamp + "_total_times.csv", header=None)
    plot(search, kind, title="Total index time in microseconds",
         ylabel='time in microseconds', graph_path=graph_path)

    # number of reads
    reads = pd.read_csv(
        predictions_path + timestamp + "_reads.csv", header=None)
    plot(reads, kind, title='Average reads required for search',
         ylabel="number of reads",graph_path=graph_path)


def plot(data, kind, title, ylabel, graph_path, binsize=50):
    """
    Draws a plot of the data.

    :param data: the data to be plotted (indexed data points)
    :param kind: "scatter" or "hist2d"
    :param title: title of the plot
    :param ylabel: label for the y axis (x is entropy level)
    :param graph_path: path at which the graph data shall be stored
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
    plt.savefig(graph_path + str(int(time.time())))
    plt.close()

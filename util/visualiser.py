import os
import config
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def show(kind: str, index: str, dataset: str, showAverage=False):
    """
    plots a view of the predicition made by an index for a data set.

    :param kind:
        type of plot to be used: "scatter" or "hist2d"
    :param index:
        the index whose prediction shall be shown.
    :param dataset:
        the data set that the prediction was made with.
    :param showAverage:
        specifies if there should be a line that shows the average of the runs

    """

    # prediction time
    pred = pd.read_csv(
        config.PREDICTIONS_PATH + "pred_times.csv", header=None)
    plot(pred, kind, title="Prediction time per key in microseconds",
         ylabel="time in microseconds")

    # search
    search = pd.read_csv(
        config.PREDICTIONS_PATH + "search_times.csv", header=None)
    plot(search, kind, title="Search time per key in microseconds",
         ylabel='time in microseconds')

    # reads
    reads = pd.read_csv(
        config.PREDICTIONS_PATH + "reads.csv", header=None)
    plot(reads, kind, title='Average reads required for search',
         ylabel="number of reads")


def plot(data, kind: str, title: str, ylabel: str, binsize=50):
    """
    Draws a plot of the data.

    :param data: the data to be plotted (indexed data points)
    :param kind: "scatter" or "hist2d"
    :param title: title of the plot
    :param ylabel: label for the y axis (x is entropy level)
    :param binsize: size of the bins for heat maps

    """
    fig, ax = plt.subplots()
    ax.set(xlabel='entropy', ylabel=ylabel, title=title)
    ax.grid()
    if kind == "scatter":
        plt.scatter(data[0], data[1])
    elif kind == "hist2d":
        plt.hist2d(data[0], data[1], bins=binsize, cmap=cm.jet)
    plt.show()


def main():
    show("hist2d", "naive_learned_index", "Integers_100x10x100k")


if __name__ == "__main__":
    config.PREDICTIONS_PATH = "../"+config.PREDICTIONS_PATH
    main()

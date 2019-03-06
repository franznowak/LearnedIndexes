import os
import config
from ..exceptions import NoPredictionForIndex, NoPredictionsForDataset


def show(index:str, dataset:str, showAverage=False, scatterRuns=True,
         scatterAll=False):
    """
    plots a view of the predicition made by an index for a data set.

    :param index:
        the index whose prediction shall be shown.
    :param dataset:
        the data set that the prediction was made with.
    :param showAverage:
        specifies if there should be a line that shows the average of the runs
    :param  scatterRuns:
        if True, each run gets its own data point(s).
    :param scatterAll:
        if True, each individual prediction gets its own data point.

    """

    if not os.path.isfile(config.PREDICTIONS_PATH+index):
        raise NoPredictionForIndex(index)
    if not os.path.isfile(config.PREDICTIONS_PATH+index+"/"+dataset):
        raise NoPredictionsForDataset(dataset)

    #
    # x = np.arange(0, config.N_KEYS, int(config.N_KEYS / config.N_SAMPLES))
    # y = np.average(array_predictions, axis=0)[6]
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set(xlabel='key', ylabel='number of reads',
    #        title='Access time array predict data 6')
    # ax.grid()
    # #plt.show()
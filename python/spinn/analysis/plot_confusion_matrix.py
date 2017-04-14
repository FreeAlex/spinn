import argparse
from report_file import load_report_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def parse_args():

    parser = argparse.ArgumentParser('Plot the attention matrix on console')
    parser.add_argument('report_file', type=str)
    parser.add_argument('--save', type=str)
    return parser.parse_args()




def plot_confusion_matrix(args):

    # construct confusion matrix
    report = load_report_file(args.report_file)

    confmat = np.zeros(shape=(3,3))
    for one in report:
        confmat[one['truth']][one['pred']] += 1

    print confmat

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confmat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_yticklabels(['', 'TE', 'TN', 'TC'])
    ax.set_xticklabels(['', 'PE', 'PN', 'PC'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # set numbers on
    for (j, i), label in np.ndenumerate(confmat):
        ax.text(i, j, int(label), ha='center', va='center')

    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()

if __name__ == '__main__':

    args = parse_args()

    plot_confusion_matrix(args)


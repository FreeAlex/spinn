import argparse
import csv
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser('Shrink ppdb data to a csv file')
    parser.add_argument('data_file', type=str)
    parser.add_argument('--plot', action='store_true')

    return parser.parse_args()


def statistic(args):

    relcount = {}
    with open(args.data_file, 'r') as csvf:
        reader = csv.DictReader(csvf)

        for row in reader:
            rel = row['relation']
            relcount[rel] = relcount.get(rel, 0) + 1

    return relcount

def plot_rel_count(relcount):

    keys = relcount.keys()
    for i in range(len(keys)):
        plt.bar(i, relcount[keys[i]], alpha=0.8)

    plt.show()


if __name__ == '__main__':

    args = parse_args()

    relcount = statistic(args)
    print relcount

    plot_rel_count(relcount)


import argparse
import random
from spinn.data.snli.load_snli_data import load_data as snli_load_data
from load_attention_matrix_log import load_attention_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def get_parser_stack(tokens, transitions):
    ti = 0
    stack = []
    parser_stack = []
    print tokens
    print transitions
    for t in transitions:
        if t == 0:
            stack.append(tokens[ti])
            parser_stack.append(tokens[ti])
            ti += 1
        else:   # t == 1
            s2 = stack.pop().split('#')[-1]  # take the second word in right
            s1 = stack.pop().split('#')[0]    # take first word in left
            stack.append('{}#{}'.format(s1, s2))
            parser_stack.append('({}##{})'.format(s1, s2))
    return parser_stack


def plot_snli_attention_matrix(args):

    examples, _ = snli_load_data(args.example_file, True)
    print 'loaded {} samples'.format(len(examples))
    atts = load_attention_matrix(args.att_file)
    index = random.randint(0, len(examples)-1) if args.index == -1 else args.index
    example = examples[index]

    id = example['example_id']

    if id not in atts:
        raise KeyError('The attention of example not exists, id: {}'.format(id))
    att = atts[id]
    matrix = np.array(att['matrix']).transpose()  # premise by hypothesis
    stackp = get_parser_stack(example['premise_tokens'], example['premise_transitions'])
    stackh = get_parser_stack(example['hypothesis_tokens'], example['hypothesis_transitions'])

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + stackh, rotation=90)
    ax.set_yticklabels([''] + stackp)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('hypothesis')
    plt.ylabel('premise')
    plt.tight_layout()
    if args.save_dir:
        plt.savefig(os.path.join(args.save_dir, '{}.png'.format(id)))
    else:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plot the attention matrix on console')
    parser.add_argument('example_file', type=str)
    parser.add_argument('att_file', type=str)
    parser.add_argument('--dataset', type=str, default='snli')
    parser.add_argument('--index', type=int, default=-1, help='the index of sentence to plot, -1: pick randomly')
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()


    if args.dataset == 'snli':
        plot_snli_attention_matrix(args)
    else:
        ValueError('dataset not supported')











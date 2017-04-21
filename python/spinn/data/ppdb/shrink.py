import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser('Shrink ppdb data to a csv file')
    parser.add_argument('data_file', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--type', type=str, default='phrasal', help='[syntax, phrasal]')
    parser.add_argument('--format', type=str, default='tsv')

    return parser.parse_args()


IDX_SOURCE = 1
IDX_TARGET = 2
IDX_RELATION = 5

def shrink_fields(args):

    with open(args.data_file, 'r') as txtfile, open(args.output, 'w') as csvfile:
        id = 0
        if args.format == 'tsv':
            writer = csv.DictWriter(csvfile, ['id', 'source', 'target', 'relation'], dialect='excel-tab')
        elif args.format == 'csv':
            writer = csv.DictWriter(csvfile, ['id', 'source', 'target', 'relation'])
        writer.writeheader()
        for row in txtfile:
            pair = row.split(' ||| ')
            writer.writerow({
                'id': id,
                'source': pair[IDX_SOURCE],
                'target': pair[IDX_TARGET],
                'relation': pair[IDX_RELATION][:-1],
            })
            id += 1

if __name__ == '__main__':

    args = parse_args()

    shrink_fields(args)


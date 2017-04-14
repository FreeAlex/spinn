import argparse
from report_file import load_report_file
from spinn.data.snli.load_snli_data import load_data as snli_load_data

def parse_args():
    parser = argparse.ArgumentParser('Gather error examples given report file')
    parser.add_argument('report_file', type=str)
    parser.add_argument('example_file', type=str)
    parser.add_argument('save', type=str)
    return parser.parse_args()


def gather_error_examples(args):

    report = load_report_file(args.report_file)
    examples, _ = snli_load_data(args.example_file, True)
    examples = {
        one['example_id']: one for one in examples
    }

    with open(args.save, 'w') as txtf:
        for one in report:
            if one['truth'] != one['pred']:
                example = examples[one['id']]
                txtf.write('{}\n'.format(one))
                txtf.write('{}\n'.format(example['premise']))
                txtf.write('{}\n'.format(example['hypothesis']))
                txtf.write('------------------------------------------------------\n')



if __name__ == '__main__':

    args = parse_args()

    gather_error_examples(args)
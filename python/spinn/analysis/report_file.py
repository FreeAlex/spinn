



def load_report_file(file):
    """

    :param file: file path
    :return: a list of dict, dict keys:
    """
    report = []
    with open(file, 'r') as txtf:
        for row in txtf:
            rowa = row.split(' ')
            report.append(
                {
                    'id': rowa[0],
                    'truth': int(rowa[2]),
                    'pred': int(rowa[3]),
                    'output': [ float(rowa[i]) for i in range(4,7)],
                }
            )

    return report
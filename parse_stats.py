import argparse
from collections import defaultdict
import re

from utils import remove_suffix


def parse_txt(stats_file):
    # assert stats_file.endswith('.txt')

    train_acc = defaultdict(lambda: defaultdict(list))
    val_acc = defaultdict(lambda: defaultdict(list))
    test_acc = defaultdict(lambda: defaultdict(list))

    with open(stats_file) as stats:
        line = stats.readline()

        while line.strip():
            m_selection = re.search('(?<=Selected )(\d+%?)', line)
            if m_selection:
                selection = m_selection.group(0)
                # m_model = re.search('(?<=: \().*(?=\))', line)
                # emsize, nhid, nlayers, nhead, activation, n_epochs = re.split('[-_]', m_model.group(0))
                # emsize, nhid, nlayers, nhead, n_epochs = int(emsize), int(nhid), int(nlayers), int(nhead), int(n_epochs)

                line = stats.readline()
                while line.strip() and not line.startswith('Selected '):
                    m_method = re.search('(.*)(?=:)', line)
                    method = m_method.group(0).lower()
                    method = remove_suffix(method, ' zen score')
                    # methods = ['lowest', 'random', 'highest', 'mixed', 'middle', 'uniformly selected', 'proposed']  # TODO: take arg
                    method = method.replace('ly selected', '').replace('mixed', 'mix')

                    line = stats.readline()
                    m_train = re.search('(?<=Train Acc: )(\d\d\.\d\d)(?=%)', line)
                    train_acc[selection][method].append(m_train.group(0))

                    line = stats.readline()
                    m_val = re.search('(?<=Val. Acc: )(\d\d\.\d\d)(?=%)', line)
                    val_acc[selection][method].append(m_val.group(0))

                    line = stats.readline()
                    m_test = re.search('(?<=Test Acc: )(\d\d\.\d\d)(?=%)', line)
                    test_acc[selection][method].append(m_test.group(0))

                    line = stats.readline()
    
    return {
        'train': train_acc,
        'val': val_acc,
        'test': test_acc
    }


def parse_csv(stats_file, sep=','):
    test_acc = defaultdict(lambda: defaultdict(list))

    import csv

    with open(stats_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for i, row in enumerate(csv_reader):
            method = row["Method"]
            scores = []
            for i in range(1, 4):
                key = f"Run {i}"
                if key in row and row[key] not in {'', '-'}:
                    scores.append(row[key])

            if "Selected #" in row and row["Selected #"] not in {'', '-'}:
                selection = row["Selected #"]          
            else:
                assert "Selected %" in row and row["Selected %"] not in {'', '-'}
                selection = row["Selected %"]

            test_acc[selection][method] += scores
    
    return test_acc


def write_table(acc_dict, sep=','):
    output_file = remove_suffix(args.stats_file, '.txt')
    if sep == ',':
        output_file += '.csv'
    elif sep == '\t':
        output_file += '.tsv'
        
    with open(output_file, 'w') as output:
        max_runs = 3  # TODO: remove this
        # TODO: write `split`?
        output.write(sep.join(["Selected #", "Selected %", "Method", "Run 1", "Run 2", "Run 3", "Average"]) + '\n')
        for selection, method_to_scores in acc_dict.items():
            for method, scores in method_to_scores.items():
                assert len(scores) <= max_runs  
                average = sum(map(float, scores)) / len(scores)
                output_line = []
                if selection.endswith('%'):
                    output_line += ['-', selection]
                else:
                    output_line += [selection, '-']
                output_line += [method] + scores + ['-'] * (max_runs - len(scores)) + [str(average)]
                output.write(sep.join(output_line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_file', type=str)
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--val', action="store_true")
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    stats_dict = parse_txt(args.stats_file)
    train_acc = stats_dict['train']
    val_acc = stats_dict['val']
    test_acc = stats_dict['test']

    if args.train:
        write_table(train_acc)
    if args.val:
        write_table(val_acc)
    if args.test:
        write_table(test_acc)

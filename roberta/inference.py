import argparse
import sys

sys.path.insert(1, '/home/VD/cychang/sijie')

from fairseq.models.roberta import RobertaModel


TASKS = ["MNLI", "QNLI", "QQP", "RTE", "SST-2", "MRPC", "CoLA", "STS-B"]

# modified `task_to_keys` from run_glue.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py#L54
TASK2KEYS = {
    "CoLA": ("sentence", None, "acceptability"),
    "MNLI": ("premise", "hypothesis", "gold_label"),
    "MRPC": ("sentence1", "sentence2", ""),  # TODO: label header
    "QNLI": ("question", "sentence", "label"),
    "QQP": ("question1", "question2", "is_duplicate"),
    "RTE": ("sentence1", "sentence2", "label"),
    "SST-2": ("sentence", None, "label"),
    "STS-B": ("sentence1", "sentence2", "score"),
    # "WNLI": ("sentence1", "sentence2", "label"),
}


def val(task, glue_data_path='/home/VD/cychang/sijie/glue_data/glue_data/'):

    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()

    dev_files = ['dev_matched', 'dev_mismatched'] if task == 'MNLI' else ['dev']

    for dev in dev_files:
        with open(f'{glue_data_path}/{task}/{dev}.tsv', encoding='utf8') as fin:
            if task == "CoLA":
                """
                https://nyu-mll.github.io/CoLA/
                Data Format

                Each line in the .tsv files consists of 4 tab-separated columns.
                Column 1:	the code representing the source of the sentence.
                Column 2:	the acceptability judgment label (0=unacceptable, 1=acceptable).
                Column 3:	the acceptability judgment as originally notated by the author.
                Column 4:	the sentence.
                """
                index1, index2, index_t = 3, None, 1
            else:
                header = fin.readline().strip().split('\t')
                index1, index_t = header.index(TASK2KEYS[task][0]), header.index(TASK2KEYS[task][2])
                if TASK2KEYS[task][1] is not None:
                    index2 = header.index(TASK2KEYS[task][1])

            for index, line in enumerate(fin):
                tokens = line.strip().split('\t')
                sent1, sent2, target = tokens[index1], tokens[index2], tokens[index_t]
                tokens = roberta.encode(sent1, sent2)
                prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
                prediction_label = label_fn(prediction)
                ncorrect += int(prediction_label == target)
                nsamples += 1

        accuracy = float(ncorrect)/float(nsamples)
        yield 'accuracy', accuracy  # TODO: implement others


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=TASKS)
    args = parser.parse_args()

    if args.task == "MRPC":
        raise NotImplementedError("dataset MRPC not correctly preprocessed")

    roberta = RobertaModel.from_pretrained(
        'checkpoints/',
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=f'/home/VD/cychang/sijie/glue_data/glue_data/{args.task}-bin'
    )

    num_classes = 3 if args.task == "MNLI" else 1 if args.task == "STS-B" else 2
    roberta.register_classification_head('sentence_classification_head', num_classes=num_classes)

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    output_filename = 'eval_results.txt'

    # TODO: change `val` function call
    if args.task == "MNLI":
        with open(output_filename, 'a+') as f:
            f.write(f'{args.task} matched\n')
        key, value = val(f'/home/VD/cychang/sijie/glue_data/glue_data/{args.task}/dev_matched.tsv')
        with open(output_filename, 'a+') as f:
            f.write(f'| {key.capitalize()}: {value}\n')

        with open(output_filename, 'a+') as f:
            f.write(f'{args.task} mismatched\n')
        key, value = val(f'/home/VD/cychang/sijie/glue_data/glue_data/{args.task}/dev_mismatched.tsv')
        with open(output_filename, 'a+') as f:
            f.write(f'| {key.capitalize()}: {value}\n')
    else:
        with open(output_filename, 'a+') as f:
            f.write(f'{args.task}\n')
        key, value = val(f'/home/VD/cychang/sijie/glue_data/glue_data/{args.task}/dev.tsv')
        with open(output_filename, 'a+') as f:
            f.write(f'| {key.capitalize()}: {value}\n')





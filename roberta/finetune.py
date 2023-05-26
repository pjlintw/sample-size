import argparse
import os


TASKS = ["MNLI", "QNLI", "QQP", "RTE", "SST-2", "MRPC", "CoLA", "STS-B"]
NUM_CLASSES = [3, 2, 2, 2, 2, 2, 2, 1]
LR = [1e-5, 1e-5, 1e-5, 2e-5, 1e-5, 1e-5, 1e-5, 2e-5]
BATCH_SIZE = [32, 32, 32, 16, 32, 16, 16, 16]
TOTAL_NUM_UPDATE = [123873, 33112, 113272, 2036, 20935, 2296, 5336, 3598]
WARMUP_UPDATES = [7432, 1986, 28318, 122, 1256, 137, 320, 214]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    args = parser.parse_args()

    i = TASKS.index(args.task)

    task = args.task
    total_num_updates = TOTAL_NUM_UPDATE[i]
    warmup_updates = WARMUP_UPDATES[i]
    lr = LR[i]
    num_classes = NUM_CLASSES[i]
    max_sentences = BATCH_SIZE[i]

    os.system(f'bash finetune.sh {total_num_updates} {warmup_updates} {lr} {num_classes} {max_sentences} {task}')


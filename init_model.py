import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torchtext.legacy import data

from sample_selector import (
    # nas.model_profiler
    epoch_time, count_parameters,
    # nas.nas_arch
    build_model, device,
    # text_classifier
    train, evaluate,
    bptt, TEXT, train_data, valid_data, test_data,
    # sample_selector
    model_arch, model_type, BATCH_SIZE,
)

from datautils import get_zen_score
from utils import int2ordinal


n_highest = 3
n_lowest = 3
n_rand = False  # TODO: change var name or implement this
flag_first_model = True  # for printing count_parameters(model)

model_paths_highest = [f'models/model_{i+1}H.pt' for i in range(n_highest)]
highest_zen_scores = [- float('inf') for i in range(n_highest)]

model_paths_lowest = [f'models/model_{i+1}L.pt' for i in range(n_lowest)]
lowest_zen_scores = [float('inf') for i in range(n_lowest)]

# if n_rand:
#     model_path_temp = 'models/model_temp.pt'
#     temp_zen_score = float('nan')


def init():
    global highest_zen_scores, lowest_zen_scores

    for i in range(n_highest):
        model_path = model_paths_highest[i]
        if os.path.isfile(model_path) and not args.overwrite_old_models:
            model = torch.load(model_path)
            if model:
                highest_zen_scores[i] = get_zen_score(model)
        else:
            print(f'init: Saving None to {model_path}')
            torch.save(None, model_path)
    # assert all(highest_zen_scores[i] >= highest_zen_scores[i+1] for i in range(n_highest - 1))

    for i in range(n_lowest):
        model_path = model_paths_lowest[i]
        if os.path.isfile(model_path) and not args.overwrite_old_models:
            model = torch.load(model_path)
            if model:
                lowest_zen_scores[i] = get_zen_score(model)
        else:
            print(f'init: Saving None to {model_path}')
            torch.save(None, model_path)
    # assert all(lowest_zen_scores[i] <= lowest_zen_scores[i+1] for i in range(n_lowest - 1))


def get_random_model():
    input_dim = len(TEXT.vocab)
    model = build_model(model_arch=model_arch, model_type=model_type, input_dim=input_dim, nlabels=1)

    global flag_first_model
    if flag_first_model:
        flag_first_model = False
        with open(args.output_file, 'a+') as f:
            f.write('\n' + '-' * 45 + '\n')
            if args.message:
                f.write(f'##### {args.message} #####\n')
            f.write(f'The model has {count_parameters(model):,} trainable parameters\n')

    return model


def save_model(model, model_path):
    print(f'Saving {"model" if model else "nothing"} to {model_path}')
    torch.save(model, model_path)


def evaluate_model(model_path, model_description, n_epochs=200):
    model = torch.load(model_path)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)  # if model_type == "transformer"

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device
    )

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_acc, ave_zen_score, zen_score = train(model_type, model, train_iter, optimizer, criterion,
                                                                src_mask)
        if epoch + 1 == n_epochs:
            valid_loss, valid_acc = evaluate(model_type, model, valid_iter, criterion, src_mask)
            test_loss, test_acc = evaluate(model_type, model, test_iter, criterion, src_mask)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Zen Score: {zen_score:5.2f}')

    with open(args.output_file, 'a+') as f:
        f.write(f'{model_description}:\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%\n')
        f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%\n')
        f.write(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%\n')


def main():
    global highest_zen_scores, lowest_zen_scores

    # if n_rand and args.repeat:
    #     rand_int = random.randrange(args.repeat)  # random index for random model init

    zen_scores_path = 'model_zen_scores.npy'
    all_zen_scores = []

    for i_run in range(args.repeat):
        model = get_random_model()
        # save_model(model, model_path_temp)
        zen_score = get_zen_score(model)
        all_zen_scores.append(zen_score)
        print(f'new zen score: {zen_score}')

        for i_h in range(n_highest):
            if zen_score > highest_zen_scores[i_h]:
                for j in range(n_highest - 1, i_h, -1):
                    highest_zen_scores[j] = highest_zen_scores[j - 1]
                    os.rename(model_paths_highest[j - 1], model_paths_highest[j])
                highest_zen_scores[i_h] = zen_score
                save_model(model, model_paths_highest[i_h])
                break

        for i_l in range(n_lowest):
            if zen_score < lowest_zen_scores[i_l]:
                for j in range(n_lowest - 1, i_l, -1):
                    lowest_zen_scores[j] = lowest_zen_scores[j - 1]
                    os.rename(model_paths_lowest[j - 1], model_paths_lowest[j])
                lowest_zen_scores[i_l] = zen_score
                save_model(model, model_paths_lowest[i_l])
                break

        # if n_rand and i_run == rand_int:
        #     global temp_zen_score
        #     temp_zen_score = zen_score
        #     save_model(model, model_path_temp)

    np.save(zen_scores_path, np.array(all_zen_scores), allow_pickle=True)

    print('Highest zen scores:')
    for i in range(n_highest):
        print(highest_zen_scores[i])
    print(f'Lowest zen scores:')
    for i in range(n_lowest):
        print(lowest_zen_scores[i])

    # print(f'Randomly picked zen score: {temp_zen_score}')

    for i in range(n_highest):
        model_path = model_paths_highest[i]
        evaluate_model(model_path, f'Model init with {f"{int2ordinal(i+1)} " if i > 0 else ""}highest zen score ({highest_zen_scores[i]})',
                       n_epochs=N_EPOCHS)
    for i in range(n_lowest):
        model_path = model_paths_lowest[i]
        evaluate_model(model_path, f'Model init with {f"{int2ordinal(i+1)} " if i > 0 else ""}lowest zen score ({lowest_zen_scores[i]})',
                       n_epochs=N_EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('repeat', nargs='?', default=10000, type=int)
    parser.add_argument('--n_highest', default=3, type=int)
    parser.add_argument('--n_lowest', default=3, type=int)
    parser.add_argument('--output_file', default='stats.txt', type=str)
    parser.add_argument('--message', default='', type=str)
    parser.add_argument('--overwrite_old_models', action='store_true')
    args = parser.parse_args()

    assert args.n_highest >= 0 and args.n_lowest >= 0

    N_EPOCHS = 100

    init()

    print('Highest zen scores:')
    for i in range(n_highest):
        print(highest_zen_scores[i])
    print(f'Lowest zen scores:')
    for i in range(n_lowest):
        print(lowest_zen_scores[i])

    main()

    # if os.path.isfile(model_path_temp):
    #     os.remove(model_path_temp)

import argparse
from pathlib import Path
import random
import time
import logging
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

# from torchtext.legacy import data

from nas.nas_arch import device, SEED, build_model
from nas.model_profiler import count_parameters, epoch_time

from datautils import train, evaluate, dataset, fit_and_predict, \
    fit_and_predict_gd_unweighted, fit_and_predict_nls_unweighted, \
    fit_and_predict_nls_weighted, fit_and_predict_gd_weighted, \
    fit_and_predict_nn, fit_and_predict_combined_parameters_minimization_methods, \
    fit_and_predict_combined_extrapolating_functions, \
    fit_and_predict_plot_mae, \
    get_zen_score, get_perplexity, \
    vocab_text, bptt, train_data, valid_data, test_data, collate_batch, num_class, output_results_filename
from utils import get_smallest_arg, get_elbow

logger = logging.getLogger(__name__)

BATCH_SIZE = 8

model_type = "transformer"

N_EPOCHS = 400  # for train_on_subset

samples_percentage = "0%"


def train_on_subset(subset, subset_name,
                    output_filename='stats.txt',
                    model_path='model_0.pt', model_type="transformer"):
    model = torch.load(model_path)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    if num_class == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    src_mask = model.generate_square_subsequent_mask(bptt).to(device)  # if model_type == "transformer"

    train_iter = DataLoader(subset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    train_loss = train_acc = valid_loss = valid_acc = test_loss = test_acc = -1
    counter_stop = 0  # for early stopping
    PATIENCE = 15
    min_train_loss = float("inf")
    warmup = 300
    counter = 0
    print("N_EPOCHS: ", N_EPOCHS)
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc, ave_zen_score, zen_score = train(model_type, model, train_iter, optimizer, criterion,
                                                                src_mask)

        if epoch + 1 == N_EPOCHS:
            test_loss, test_acc = evaluate(model_type, model, test_iter, criterion, src_mask)

        if epoch > warmup:
            if train_loss >= min_train_loss and epoch + 1 >= 2:
                counter_stop += 1
            else:
                counter_stop = 0
            if counter_stop > PATIENCE:
                test_loss, test_acc = evaluate(model_type, model, test_iter, criterion, src_mask)
                break

        if train_loss < min_train_loss:
            min_train_loss = train_loss

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(
            f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Early Stopping Counter {counter_stop}')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | 1st Zen Score: {zen_score:5.5f} | Ave. Zen Score: {ave_zen_score:5.5f}')

    with open(output_filename, 'a+') as f:
        f.write(f'{subset_name.capitalize()}:\n')
        f.write(f'# Stopped at Epoch {epoch + 1}\n')
        f.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%\n')
        # f.write(f'\tVal. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%\n')
        f.write(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%\n')

    return test_acc


def get_sample_zen_scores(train_samples, print_all=False):
    zen_scores = []

    start_time = time.time()

    for i, train_sample in enumerate(train_samples):
        model = torch.load(model_path)
        sample_zen_score = get_zen_score(model, [train_sample])

        zen_scores.append(sample_zen_score)
        if print_all:
            print("sample_zen_score: ", sample_zen_score)
        else:
            if len(zen_scores) % 1000 == 1 or len(zen_scores) == len(train_data):
                print(f'{len(zen_scores)}/{len(train_data)} zen scores in {time.time() - start_time}')
    return zen_scores


def main():
    try:
        print(f"number of epochs: {N_EPOCHS}")
        model = torch.load(model_path)
        print('Model loaded')
    except FileNotFoundError:
        print('Initializing model...')
        input_dim = len(vocab_text)

        if num_class == 2:
            nlabels = 1
        else:
            nlabels = num_class

        model = build_model(model_arch=model_arch, model_type=model_type, input_dim=input_dim, nlabels=nlabels,
                            activation=activation)
        print(f'Saving model to {model_path}')
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    zen_scores_path = f'test/sample_zen_scores_({emsize}-{nhid}-{nlayers}-{nhead}_{activation})_{dataset}'

    try:
        loaded = np.load(zen_scores_path + ".npz")
        zen_scores = loaded["a"]
        print('zen_scores loaded')
    except FileNotFoundError:
        zen_scores = get_sample_zen_scores(train_samples=train_data)

        print(f'Saving zen scores to {zen_scores_path}')
        print("Type of zen_scores: ", type(zen_scores))
        Path(zen_scores_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(zen_scores_path, a=np.array(zen_scores))

    sorted_indices = np.argsort(zen_scores)  # ascending
    # assert len(sorted_indices) == len(train_data)

    output_filename = args.output_file
    print("output_filename: ", output_filename)
    if args.clear_output:
        with open(output_filename, 'w'):
            pass

    if calculate_perplexity:
        try:
            loaded_ppl_scores = np.load(f'perplexity_scores/{dataset}_ppl_scores' + ".npz")
            ppl_scores_array = loaded_ppl_scores["a"]
            print(f'ppl_scores loaded for {dataset}')
        except FileNotFoundError:
            print('calculating ppl_scores')
            ppl_scores_array = get_perplexity()
            print(f'ppl_scores loaded for {dataset}')
            np.savez_compressed(f'perplexity_scores/{dataset}_ppl_scores', a=ppl_scores_array)

        sorted_ppl_scores_indices = np.argsort(ppl_scores_array)

    train_accuracy = []
    sample_sizes = []

    for selection in args.train_size:
        selection_log = selection if selection.endswith('%') else f'{selection} samples'
        if selection.endswith('%'):
            pct = int(selection.rstrip('%'))
            num_samples = int(pct / 100 * len(train_data))
            selection_log = f'{pct}%'
            print("PCT: ", pct)
            print("num_samples: ", num_samples)
            sample_sizes.append(num_samples)
        else:
            num_samples = int(selection)
            selection_log = f'{num_samples} samples'
            sample_sizes.append(num_samples)

        with open(output_filename, 'a+') as f:

            f.write(f'Selected {selection_log}: ({emsize}-{nhid}-{nlayers}-{nhead}_{activation}_{N_EPOCHS})\n')

        if train_model and sampling_method == "random":
            sample_accuracy = []
            iter_per_split = 1

            for i in range(iter_per_split):
                # select m points
                random_subset = [train_data[i] for i in random.sample(range(len(train_data)), num_samples)]
                accuracy = train_on_subset(random_subset, 'random zen score', output_filename=output_filename,
                                           model_path=model_path, model_type=model_type)
                sample_accuracy.append(accuracy)
                print(f"iter_per_split completed: {i}")

            train_accuracy.append(sample_accuracy)

        if train_model and sampling_method == "lowest":
            sample_accuracy = []
            # select lowest points
            lowest_subset = [train_data[i] for i in sorted_ppl_scores_indices[:num_samples]]
            accuracy = train_on_subset(lowest_subset, 'lowest zen score', output_filename=output_filename, model_path=model_path, model_type=model_type)
            sample_accuracy.append(accuracy)

            train_accuracy.append(sample_accuracy)

        print("COMPLETED")

    if not train_model:
        if sampling_method == "random":
            # 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100%
            if dataset == "IMDB":
                train_accuracy = [
                    [0.5], [0.5], [0.51172], [0.53608], [0.55315], [0.55828], [0.57256], [0.58348], [0.61404], [0.6266],
                    [0.654], [0.67676], [0.70672],
                    [0.71932, 0.71312, 0.70428],
                    [0.73344], [0.72252], [0.73836], [0.73004], [0.73164],
                    [0.7428],
                    [0.74344], [0.75008], [0.75308], [0.75422], [0.75984], [0.76428], [0.7594], [0.75792], [0.76752]
                ]

            elif dataset == "SST2":
                train_accuracy = [
                    [0.53325], [0.57454], [0.61157], [0.64678], [0.68922], [0.69839], [0.68922], [0.71559], [0.71330],
                    [0.70183],
                    [0.69266], [0.77064], [0.74197], [0.74885], [0.79587], [0.77293], [0.79243], [0.78440], [0.79816],
                    [0.79128],
                    [0.80275], [0.77866], [0.80619], [0.80160], [0.81192], [0.78211], [0.79128], [0.80170], [0.81766]
                ]

            elif dataset == "AG_NEWS":
                train_accuracy = [
                    [0.42282], [0.54907], [0.59486], [0.65802], [0.69328], [0.71334], [0.70853], [0.72082], [0.73312],
                    [0.75924],
                    [0.81289], [0.83578], [0.83657], [0.84973], [0.85105], [0.85894], [0.86697], [0.86907], [0.87955],
                    [0.88531],
                    [0.88197], [0.88513], [0.88776], [0.88355], [0.88815], [0.89210], [0.88368], [0.89078], [0.90276]
                ]

            elif dataset == "DBpedia":
                train_accuracy = [
                    [0.33501], [0.49844], [0.54442], [0.59617], [0.67137], [0.65514], [0.67044], [0.68894], [0.69932],
                    [0.7248],
                    [0.75837], [0.7916], [0.78308], [0.80917], [0.81941], [0.82662], [0.83068], [0.83225], [0.83787],
                    [0.84182],
                    [0.84317], [0.84585], [0.8459], [0.85071], [0.84937], [0.85155], [0.85241], [0.85644], [0.85678]
                ]

        elif sampling_method == "lowest":
            # 1% 2% 3% 4% 5% 6% 7% 8% 9% 10% 13% 17% 21% 25% 28% 34% 40% 43% 47% 50% 55% 60% 65% 70% 80% 85% 90% 95% 100%
            if dataset == "IMDB":
                train_accuracy = [
                    [0.5], [0.50704], [0.51864], [0.53952], [0.55068], [0.57556], [0.56584], [0.59608], [0.61342], [0.62396],
                    [0.65038], [0.68612], [0.70416], [0.71556],
                    [0.718], [0.71224], [0.72072], [0.72084], [0.72158],
                    [0.73581],
                    [0.73516], [0.73348], [0.73859], [0.75036], [0.75072], [0.75876], [0.75329], [0.76476], [0.76996]
                ]

            elif dataset == "SST2":
                train_accuracy = [
                    [0.54816], [0.58371], [0.61123], [0.65289], [0.66492], [0.67534], [0.68824], [0.68940], [0.68844], [0.69857],
                    [0.72133], [0.73821], [0.75344], [0.76032],
                    [0.76490], [0.77637], [0.76344], [0.76776], [0.76114],
                    [0.78584],
                    [0.78915], [0.77421], [0.78261], [0.77752], [0.79701], [0.79641], [0.81142], [0.79357], [0.82142]
                ]

            elif dataset == "AG_NEWS":
                train_accuracy = [
                    [0.44589], [0.58263], [0.62684], [0.66253], [0.69186], [0.69512], [0.71063], [0.72276], [0.73118],
                    [0.73934],
                    [0.74960], [0.77473], [0.79565], [0.81131], [0.81828], [0.82921], [0.84139], [0.85236], [0.85956],
                    [0.87105],
                    [0.87526], [0.87789], [0.88171], [0.88776], [0.89313], [0.89026], [0.89578], [0.89689], [0.90451]
                ]

            elif dataset == "DBpedia":
                train_accuracy = [
                    [0.37483], [0.50192], [0.55712], [0.60854], [0.63793], [0.65831], [0.67847], [0.68982], [0.69932],
                    [0.71511],
                    [0.74544], [0.7591], [0.76065], [0.7757], [0.79435], [0.80827], [0.82411], [0.82758], [0.82622],
                    [0.82725],
                    [0.83581], [0.83704], [0.84068], [0.84441], [0.84732], [0.84187], [0.8433], [0.84109], [0.8452]
                ]

    print("datatset: ", dataset)
    print("train_accuracy: ", train_accuracy)
    print("sample_sizes: ", sample_sizes)

    with open('results/accuracy_and_sampleSize_' + str(
            N_EPOCHS) + 'epochs_' + dataset + '_dataset_' + training_accuracy_filename + '.txt', 'a+') as f:
        f.write(f'sampling_method:\n')
        f.write(f'{sampling_method}.\n\n')
        f.write(f'train_accuracy:\n')
        f.write(f'{train_accuracy}.\n\n')
        f.write(f'sample_sizes:\n')
        f.write(f'{sample_sizes}.\n\n')

    if plot_minimization_methods:
        fit_and_predict_combined_parameters_minimization_methods(train_accuracy,
                                                                 sample_sizes,
                                                                 int(len(train_data)),
                                                                 N_EPOCHS, minimization_methods_extrapolating_func,
                                                                 show_percentage_labels, random_init, seed_option)

    if plot_extrapolating_functions:
        fit_and_predict_combined_extrapolating_functions(train_accuracy, sample_sizes, int(len(train_data)),
                                                         extrapolating_training_index, show_percentage_labels, sampling_method)


    if plot_mae:
        fit_and_predict_plot_mae(train_accuracy, sample_sizes, show_percentage_labels, sampling_method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_size', nargs='*', type=str)
    parser.add_argument('--nhid', default=1820, type=int)
    parser.add_argument('--nlayers', default=2, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_epochs', default=10000, type=int)
    parser.add_argument('--train_model', default=False, type=bool)
    parser.add_argument('--plot_minimization_methods', default=False, type=bool)
    parser.add_argument('--minimization_methods_extrapolating_func', default='exp', type=str)
    parser.add_argument('--plot_extrapolating', default=False, type=bool)
    parser.add_argument('--extrapolating_training_index', default=20, type=int)
    parser.add_argument('--plot_mae', default=False, type=bool)
    parser.add_argument('--perplexity', default=False, type=bool)
    parser.add_argument('--sampling', default='random', type=str)
    parser.add_argument('--show_percentage_labels', default=False, type=bool)
    parser.add_argument('--random_init', default=False, type=bool)
    parser.add_argument('--seed_option', default=1, type=int)
    parser.add_argument('--training_accuracy_filename', default='9999', type=str)
    parser.add_argument('--output_file', default='results/test.txt', type=str)
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument("--clear_output",
                        action='store_true',
                        help="Whether to clear output stats txt file")
    args = parser.parse_args()

    emsize = 1000
    N_EPOCHS = args.n_epochs
    nhid = args.nhid
    nlayers = args.nlayers
    BATCH_SIZE = args.batch_size
    nhead = 4
    plot_extrapolating_functions = args.plot_extrapolating
    extrapolating_training_index = args.extrapolating_training_index
    plot_mae = args.plot_mae
    plot_minimization_methods = args.plot_minimization_methods
    minimization_methods_extrapolating_func = args.minimization_methods_extrapolating_func
    calculate_perplexity = args.perplexity
    sampling_method = args.sampling
    random_init = args.random_init
    seed_option = args.seed_option
    show_percentage_labels = args.show_percentage_labels
    train_model = args.train_model
    model_arch = {
        'emsize': emsize,
        'nhid': nhid,
        'nlayers': nlayers,
        'nhead': nhead,
    }
    training_accuracy_filename = args.training_accuracy_filename
    print(f"training_accuracy_filename: {training_accuracy_filename}")
    activation = args.activation.lower()
    model_path = f'models/model_({emsize}-{nhid}-{nlayers}-{nhead}_{activation})_{dataset}.pt'

    for selection in args.train_size:
        if selection.endswith('%'):
            samples_percentage = selection
            pct = int(selection.rstrip('%'))
            print("PCT: ", pct)
            if pct < 0 or pct > 100:
                raise ValueError(f'Unable to select {pct}%')
        else:
            num = int(selection)
            if num < 0 or num > len(train_data):
                raise ValueError(f'Unable to select {num} samples')

    main()

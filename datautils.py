import random

import torch
from torch import nn, optim
from torchtext.datasets import IMDB, SST2, AG_NEWS, DBpedia
from sklearn.model_selection import train_test_split
import torchdata
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from itertools import chain

from nas.nas_arch import device, SEED
from utils import binary_accuracy, multi_class_accuracy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm


dataset = "IMDB"
tokenizer = get_tokenizer('spacy', "en_core_web_sm")

if dataset == "IMDB":
    bptt = 100
    train_iter_split, test_iter_split = IMDB(split=('train', 'test'))
    num_class = len(set([label for (label, text) in train_iter_split]))
elif dataset == "SST2":
    bptt = 20
    train_iter_split, test_iter_split = SST2(split=('train', 'dev'))
    num_class = len(set([label for (text, label) in train_iter_split]))
elif dataset == "AG_NEWS":
    bptt = 20
    train_iter_split, test_iter_split = AG_NEWS(split=('train', 'test'))
    num_class = len(set([label for (label, text) in train_iter_split]))
elif dataset == "DBpedia":
    bptt = 10
    train_iter_split, test_iter_split = DBpedia(split=('train', 'test'))
    num_class = len(set([label for (label, text) in train_iter_split]))

MAX_VOCAB_SIZE = 30522

print(f'dataset: {dataset}')
print(f'bptt: {bptt}')
print(f'Number of classes: {num_class}')


# build vocab
def yield_tokens_text(data_iter):
    for text_one, text_two in data_iter:
        if dataset == "IMDB" or dataset == "AG_NEWS" or dataset == "DBpedia":
            yield tokenizer(text_two)
        elif dataset == "SST2":
            yield tokenizer(text_one)
        else:
            return None


def yield_tokens_label(data_iter):
    for label, _ in data_iter:
        yield tokenizer(label)


vocab_text = build_vocab_from_iterator(yield_tokens_text(train_iter_split), max_tokens=MAX_VOCAB_SIZE,
                                       specials=["<unk>"])
vocab_text.set_default_index(vocab_text["<unk>"])

if dataset == "IMDB":
    vocab_label = build_vocab_from_iterator(yield_tokens_label(train_iter_split))
    label_pipeline = lambda x: vocab_label(tokenizer(x))

train_data_dataset = to_map_style_dataset(train_iter_split)
test_data = to_map_style_dataset(test_iter_split)
num_train = int(len(train_data_dataset) * 0.70)
train_data, valid_data = random_split(train_data_dataset, [num_train, len(train_data_dataset) - num_train])

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(f"Unique tokens in TEXT vocabulary: {len(vocab_text)}")
if dataset == "IMDB":
    print(f"Unique tokens in LABEL vocabulary: {len(vocab_label)}")

BATCH_SIZE = 64
output_results_filename = 'results/accuracy_and_mae_'

text_pipeline = lambda x: to_2d_list(vocab_text(tokenizer(x)))

model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained('gpt2')
#max_length_ppl = model_gpt2.config.n_positions
#stride = 128
max_length_ppl = 128
stride = 16

def to_2d_list(list_1d):
    return [list_1d[i:i + 1] for i in range(0, len(list_1d), 1)]


def collate_batch_imdb(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if len(processed_text) < bptt:
            torch_ones = torch.ones(bptt - len(processed_text), 1, dtype=torch.int64)
            processed_text = torch.cat((processed_text, torch_ones), 0)
        processed_text = processed_text[:bptt]
        text_list.append(processed_text)
    label_list = list(chain.from_iterable(label_list))
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.cat(text_list, 1)
    return label_list.to(device), text_list.to(device)


def collate_batch_sst2(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if len(processed_text) < bptt:
            torch_ones = torch.ones(bptt - len(processed_text), 1, dtype=torch.int64)
            processed_text = torch.cat((processed_text, torch_ones), 0)
        processed_text = processed_text[:bptt]
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.cat(text_list, 1)
    return label_list.to(device), text_list.to(device)


def collate_batch_ag_news_and_dbpedia(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(_label - 1)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if len(processed_text) < bptt:
            torch_ones = torch.ones(bptt - len(processed_text), 1, dtype=torch.int64)
            processed_text = torch.cat((processed_text, torch_ones), 0)
        processed_text = processed_text[:bptt]
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.long)
    text_list = torch.cat(text_list, 1)
    return label_list.to(device), text_list.to(device)


def calculate_perplexity(text):
  nlls = []
  prev_end_loc = 0
  encodings = tokenizer_gpt2(text, return_tensors="pt")
  seq_len = encodings["input_ids"].size(1)

  for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length_ppl, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model_gpt2(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over input tokens.
        # Multiply it with trg_len to get the summation instead of average.
        # We will take average over all the tokens to get the true average
        # in the last step of this example.
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

  ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
  print(ppl)
  return ppl


def collate_batch_perplexity_imdb_ag_news_and_dbpedia(batch):
    ppl_list = []
    for (_label, _text) in batch:
         text_ppl = calculate_perplexity(_text)
         ppl_list.append(text_ppl.item())
    return ppl_list

def collate_batch_perplexity_sst2(batch):
    ppl_list = []
    for (_text, _label) in batch:
         text_ppl = calculate_perplexity(_text)
         ppl_list.append(text_ppl.item())
    return ppl_list


if dataset == "IMDB":
    collate_batch = collate_batch_imdb
    collate_batch_perplexity = collate_batch_perplexity_imdb_ag_news_and_dbpedia
elif dataset == "SST2":
    collate_batch = collate_batch_sst2
    collate_batch_perplexity = collate_batch_perplexity_sst2
elif dataset == "AG_NEWS" or dataset == "DBpedia":
    collate_batch = collate_batch_ag_news_and_dbpedia
    collate_batch_perplexity = collate_batch_perplexity_imdb_ag_news_and_dbpedia


def get_perplexity():
    train_iter_perplexity = DataLoader(train_data, batch_size=len(train_data), collate_fn=collate_batch_perplexity)
    ppl_imdb = []
    for i, (ppl_list) in enumerate(train_iter_perplexity):
        ppl_imdb = ppl_list

    imdb_ppl_scores_array = np.array(ppl_imdb)
    return imdb_ppl_scores_array


def train(model_type, model, iterator, optimizer, criterion, src_mask):
    """
    from text_classifier
    """
    epoch_loss, epoch_acc, zen_scores, first_zen_score = 0, 0, 0, 0
    model.train()
    for i, (label, text) in enumerate(iterator):
        optimizer.zero_grad()
        assert model_type == "transformer"
        predictions, zen_score = model(text, src_mask, tgt=None, is_zen=False)
        loss = criterion(predictions, label)
        if num_class == 2:
            acc = binary_accuracy(predictions, label)
        else:
            acc = multi_class_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        zen_scores += zen_score.item()
        if i == 0:
            first_zen_score = zen_score

    # TODO fix average zen scores
    return epoch_loss / len(iterator), epoch_acc / len(iterator), zen_scores / len(iterator), first_zen_score


def evaluate(model_type, model, iterator, criterion, src_mask):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for i, (label, text) in enumerate(iterator):
            assert model_type == "transformer"
            predictions, zen_score = model(text, src_mask, tgt=None, is_zen=False)
            loss = criterion(predictions, label)
            if num_class == 2:
                acc = binary_accuracy(predictions, label)
            else:
                acc = multi_class_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def get_zen_score(model, train_subset=train_data):
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    if num_class == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # assert model_type == "transformer"
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    train_iter = DataLoader(train_subset, batch_size=1, collate_fn=collate_batch)
    valid_iter = DataLoader(valid_data, batch_size=1, collate_fn=collate_batch)
    test_iter = DataLoader(test_data, batch_size=1, collate_fn=collate_batch)

    epoch_loss, epoch_acc, zen_scores, first_zen_score = 0, 0, 0, 0
    model.train()
    for i, (label, text) in enumerate(train_iter):
        optimizer.zero_grad()

        # assert model_type == "transformer"
        predictions, zen_score = model(text, src_mask, tgt=None, is_zen=True)
        if num_class > 2:
            predictions = torch.unsqueeze(predictions, 0)
        loss = criterion(predictions, label)
        if num_class == 2:
            acc = binary_accuracy(predictions, label)
        else:
            acc = multi_class_accuracy(predictions, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        zen_scores += zen_score.item()
        if i == 0:
            first_zen_score = zen_score
            break

    return first_zen_score.item()


def fit_and_predict_nn(train_acc, sample_sizes, pred_sample_size):
    training_indexes = [0, 2, 6, 16]

    sample_sizes_training = [sample_sizes[i] for i in training_indexes]
    accuracies_training = [train_acc[i] for i in training_indexes]
    mean_acc_training = [np.mean(i) for i in accuracies_training]

    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]

    sample_sizes_array = np.array(sample_sizes_training)
    mean_acc_array = np.array(mean_acc_training)

    def exp_func(x, a, b):
        return a * (x ** b)

    def inverse_power_law_func(x, c, d, e):
        return (1 - c) - d * (x ** e)

    def power4_func(x, a, b, c, d):
        return a - (((b * x) + c) ** (-d))

    popt, _ = curve_fit(exp_func, sample_sizes_array, mean_acc_array, maxfev=1000, bounds=(-7, 7))
    a, b = popt

    popt, _ = curve_fit(inverse_power_law_func, sample_sizes_array, mean_acc_array, maxfev=1000, bounds=(-7, 7))
    c, d, e = popt

    popt, _ = curve_fit(power4_func, sample_sizes_array, mean_acc_array, method='dogbox', maxfev=5000)
    f, g, h, i = popt

    print(f"Curve fit weights exp func: a = {a} and b = {b}.")
    print(f"Curve fit weights inverse power law func: a = {c}, b = {d} and c = {e}.")
    print(f"Curve fit weights power4 func: a = {f}, b = {g}, c = {h} and d = {i}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc_exp_func = exp_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b).detach().numpy()
    max_acc_inverse_power_law_func = inverse_power_law_func(torch.tensor(pred_sample_size, dtype=torch.float32), c, d,
                                                            e).detach().numpy()
    max_acc_power4_func = power4_func(torch.tensor(pred_sample_size, dtype=torch.float32), f, g, h,
                                      i).detach().numpy()
    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc_exp_func} is predicted for {pred_sample_size} samples using exp func.")
    print(
        f"A model accuracy of {max_acc_inverse_power_law_func} is predicted for {pred_sample_size} samples using inverse power law func.")
    print(f"A model accuracy of {max_acc_power4_func} is predicted for {pred_sample_size} samples using power4 func.")

    # sample_sizes_training = np.linspace(1, pred_sample_size, pred_sample_size)

    sample_sizes_training_exp = np.linspace(1, 5834, 5834)
    sample_sizes_training_inv = np.linspace(5835, 11667, 5833)
    sample_sizes_training_pow4 = np.linspace(11668, 17500, 5833)

    print(f"Sample sizes training exp = {sample_sizes_training_exp}")
    print(f"Sample sizes training inv = {sample_sizes_training_inv}")
    print(f"Sample sizes training pow4 = {sample_sizes_training_pow4}")

    # sample_sizes_training = np.random.choice(17500, 8500, replace=False)
    accuracy_training_exp_func = exp_func(torch.tensor(sample_sizes_training_exp, dtype=torch.float32), a,
                                          b).detach().numpy()
    accuracy_training_inv_func = inverse_power_law_func(torch.tensor(sample_sizes_training_inv, dtype=torch.float32), c,
                                                        d, e).detach().numpy()
    accuracy_training_pow4_func = power4_func(torch.tensor(sample_sizes_training_pow4, dtype=torch.float32), f, g, h,
                                              i).detach().numpy()

    print(f"Length of sample_sizes_training exp = {len(accuracy_training_exp_func)}")
    print(f"Length of sample_sizes_training inv = {len(accuracy_training_inv_func)}")
    print(f"Length of sample_sizes_training power4 = {len(accuracy_training_pow4_func)}")

    print(f"Accuracy training exp = {accuracy_training_exp_func}")
    print(f"Accuracy training inv = {accuracy_training_inv_func}")
    print(f"Accuracy training pow4 = {accuracy_training_pow4_func}")

    sample_sizes_mae = list(sample_sizes)
    mean_mae = list(mean_acc)

    for index in sorted(training_indexes, reverse=True):
        del sample_sizes_mae[index]
        del mean_mae[index]

    mae = nn.L1Loss()
    output_exp = mae(exp_func(torch.tensor(sample_sizes_mae, dtype=torch.float32), a, b),
                     torch.tensor(mean_mae, dtype=torch.float32))
    output_inverse = mae(inverse_power_law_func(torch.tensor(sample_sizes_mae, dtype=torch.float32), c, d, e),
                         torch.tensor(mean_mae, dtype=torch.float32))
    output_pow4 = mae(power4_func(torch.tensor(sample_sizes_mae, dtype=torch.float32), f, g, h, i),
                      torch.tensor(mean_mae, dtype=torch.float32))
    print(f"The mae for the exp curve fit is {output_exp}.")
    print(f"The mae for the inverse curve fit is {output_inverse}.")
    print(f"The mae for the power4 curve fit is {output_pow4}.")

    # sample_sizes_training = sample_sizes_training.reshape((17500, 1))

    sample_sizes_training_exp = sample_sizes_training_exp.reshape((5834, 1))
    sample_sizes_training_inv = sample_sizes_training_inv.reshape((5833, 1))
    sample_sizes_training_pow4 = sample_sizes_training_pow4.reshape((5833, 1))

    accuracy_training_exp_func = accuracy_training_exp_func.reshape((5834, 1))
    accuracy_training_inv_func = accuracy_training_inv_func.reshape((5833, 1))
    accuracy_training_pow4_func = accuracy_training_pow4_func.reshape((5833, 1))

    flag_exp = np.zeros((17500, 1), dtype=int)
    flag_inv = np.ones((17500, 1), dtype=int)

    horizontal_stack_exp = np.hstack((sample_sizes_training_exp, accuracy_training_exp_func))
    horizontal_stack_inv = np.hstack((sample_sizes_training_inv, accuracy_training_inv_func))
    horizontal_stack_pow4 = np.hstack((sample_sizes_training_pow4, accuracy_training_pow4_func))

    final_stack = np.vstack((horizontal_stack_exp, horizontal_stack_inv, horizontal_stack_pow4))

    # final_stack = np.hstack((sample_sizes_training, accuracy_training_exp_func))

    np.random.shuffle(final_stack)

    print(f"final stack shape = {final_stack.shape}")

    x_train, x_test, y_train, y_test = train_test_split(final_stack[:, 0:1], final_stack[:, 1], test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    print(f"x_train shape = {x_train.shape}")
    print(f"y_train shape = {y_train.shape}")
    print(f"x_val shape = {x_val.shape}")
    print(f"y_val shape = {y_val.shape}")
    print(f"x_test shape = {x_test.shape}")
    print(f"y_test shape = {y_test.shape}")

    y_train = y_train.reshape((int(len(y_train)), 1))
    y_val = y_val.reshape((int(len(y_val)), 1))
    y_test = y_test.reshape((int(len(y_test)), 1))

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    print('Initializing Learning Curve Model...')
    # define model architecture
    model = nn.Sequential(
        nn.Linear(1, 12),
        nn.ReLU(),
        nn.Linear(12, 1),
        nn.Sigmoid()
    )

    path = "./lc_model/lc_model.pth"

    try:
        model.load_state_dict(torch.load(path))
        print('Learning Curve Model Weights loaded')
    except FileNotFoundError:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        train_model(model, x_train, y_train, criterion, optimizer, 1000)
        model.load_state_dict(torch.load(path))

    model.eval()
    val_model(model, x_val, y_val, criterion)
    val_model(model, x_test, y_test, criterion)
    input_data = [[15000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))
    input_data = [[25000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))
    input_data = [[50000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))
    input_data = [[80000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))
    input_data = [[100000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))
    input_data = [[200000]]
    x_data = torch.tensor(input_data).float()
    print(model(x_data))


def val_model(model, x_val, y_val, loss_fn, path="./lc_model/lc_model.pth"):
    inputs = torch.from_numpy(x_val).float()
    labels = torch.from_numpy(y_val).float()
    # model.load_state_dict(torch.load(path))
    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        print("Loss: ", loss.item())


def train_model(model, x_train, y_train, loss_fn, optimizer, epochs, path="./lc_model/lc_model.pth"):
    inputs = torch.from_numpy(x_train).float()
    labels = torch.from_numpy(y_train).float()
    epoch_loss = 100000
    for epoch in range(epochs):
        # Compute prediction and loss
        # inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        if loss.item() < epoch_loss:
            torch.save(model.state_dict(), path)
            epoch_loss = loss.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("training finished")


def fit_and_predict_plot_mae(train_acc, sample_sizes, show_percentage_labels, sampling_method):
    slicing_index = 20

    exp_mae = []
    inv_mae = []
    pow4_mae = []
    comb_mae = []

    testing_sample_sizes = sample_sizes[slicing_index:len(sample_sizes)]
    testing_accuracies = train_acc[slicing_index:len(train_acc)]
    testing_mean_acc = [np.mean(i) for i in testing_accuracies]

    print(f"testing_sample_sizes: {testing_sample_sizes}")
    print(f"testing_accuracies: {testing_accuracies}")

    # Define variables for learning rate and number of epochs for fitting with TF
    if dataset == "IMDB":
        learning_rate = 0.01
    elif dataset == "SST2" or dataset == "AG_NEWS" or dataset == "DBpedia":
        learning_rate = 0.001

    training_epochs = 5000
    mse = nn.MSELoss()

    for si in range(slicing_index):
        training_sample_sizes = sample_sizes[:si + 1]
        training_accuracies = train_acc[:si + 1]

        print(f"training_sample_sizes: {training_sample_sizes}")
        print(f"training_accuracies: {training_accuracies}")

        training_mean_acc = [np.mean(i) for i in training_accuracies]

        training_sample_sizes_array = np.array(training_sample_sizes)
        training_mean_acc_array = np.array(training_mean_acc)

        def exp_func(x, a, b):
            return a * (x ** b)

        def inverse_power_law_func(x, a, b, c):
            return (1 - a) - b * (x ** c)

        def power4_func(x, a, b, c, d):
            return a - (((b * x) + c) ** (-d))

        def exp_inv_power4_combined_func(x, a, b, c, d, e, f, g, h, i):
            return (a * (x ** b)) + ((1 - c) - d * (x ** e)) + (f - (((g * x) + h) ** (-i)))

        def custom_loss_function(y_pred, y_true, sigma):
            squared_difference = ((y_pred - y_true) ** 2) * sigma
            loss = torch.mean(squared_difference)
            return loss

        sigma = np.ones(len(training_sample_sizes))

        start_index = len(sigma)
        stop_index = 0
        step = -1
        count = 0

        # applying for loop
        for i in range(start_index, stop_index, step):
            sigma[count] = i / len(sigma)
            count = count + 1

        popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array, bounds=(-7, 7),
                            sigma=sigma, maxfev=1000000)
        a, b = popt

        popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array,
                            sigma=sigma, bounds=(-7, 7), maxfev=1000000)
        c, d, e = popt

        if dataset == "SST2":
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                sigma=sigma, maxfev=10000000)
            f, g, h, i = popt
        else:
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                sigma=sigma, maxfev=1000000)
            f, g, h, i = popt

        sigma_ensemble = np.ones(len(training_sample_sizes))
        start_index = 0
        stop_index = len(sigma)
        step = 1

        # applying for loop
        for index in range(start_index, stop_index, step):
            sigma_ensemble[index] = (index + 1) / len(sigma_ensemble)

        j = torch.tensor(0.0, requires_grad=True)
        k = torch.tensor(0.0, requires_grad=True)
        l = torch.tensor(0.0, requires_grad=True)
        m = torch.tensor(0.0, requires_grad=True)
        n = torch.tensor(0.0, requires_grad=True)
        o = torch.tensor(0.0, requires_grad=True)
        p = torch.tensor(0.0, requires_grad=True)
        q = torch.tensor(0.0, requires_grad=True)
        r = torch.tensor(0.0, requires_grad=True)

        for epoch in range(training_epochs):
            y_pred = exp_inv_power4_combined_func(torch.tensor(training_sample_sizes, dtype=torch.float32), j, k, l, m,
                                                  n, o, p, q, r)
            cost_function = mse(y_pred, torch.tensor(training_mean_acc, dtype=torch.float32))
            # cost_function = custom_loss_function(y_pred, torch.tensor(training_mean_acc, dtype=torch.float32), torch.tensor(sigma_ensemble, dtype=torch.float32))
            cost_function.backward()
            with torch.no_grad():
                j -= j.grad * learning_rate
                k -= k.grad * learning_rate
                l -= l.grad * learning_rate
                m -= m.grad * learning_rate
                n -= n.grad * learning_rate
                o -= o.grad * learning_rate
                p -= p.grad * learning_rate
                q -= q.grad * learning_rate
                r -= r.grad * learning_rate
                j.grad.zero_()
                k.grad.zero_()
                l.grad.zero_()
                m.grad.zero_()
                n.grad.zero_()
                o.grad.zero_()
                p.grad.zero_()
                q.grad.zero_()
                r.grad.zero_()

        # print(f"Curve fit weights exp func with weighted nls: a = {a} and b = {b}.")
        # print(f"Curve fit weights inverse power law func with weighted nls: a = {c}, b = {d} and c = {e}.")
        # print(f"Curve fit weights power4 func with weighted nls: a = {f}, b = {g}, c = {h} and d = {i}.")
        # print(f"Curve fit weights ensemble func with unweighted gd: a = {j}, b = {k}, c = {l}, d = {m}, e = {n}, f = {o}, g= {p}, h = {q} and i = {r}.")

        mae = nn.L1Loss()

        output_exp = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), a, b),
                         torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_inv = mae(inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), c, d, e),
                         torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_pow4 = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), f, g, h, i),
                          torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_comb = mae(
            exp_inv_power4_combined_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), j, k, l, m, n, o, p,
                                         q, r),
            torch.tensor(testing_mean_acc, dtype=torch.float32))

        # print(f"The mae for the exp func curve fit is {output_exp} using unweighted nls.")
        # print(f"The mae for the inverse power law func curve fit is {output_inv} using unweighted nls.")
        # print(f"The mae for the power4 func curve fit is {output_pow4} using unweighted nls.")
        exp_mae.append(output_exp.item())
        inv_mae.append(output_inv.item())
        pow4_mae.append(output_pow4.item())
        comb_mae.append(output_comb.item())
    print(exp_mae)
    print(inv_mae)
    print(pow4_mae)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.array(sample_sizes[:len(exp_mae)]),
            np.array(exp_mae),
            color='red', label="Exp")

    ax.plot(np.array(sample_sizes[:len(inv_mae)]),
            np.array(inv_mae),
            color='blue', label="Inverse")

    ax.plot(np.array(sample_sizes[:len(pow4_mae)]),
            np.array(pow4_mae),
            color='green', label="Pow4")

    ax.plot(np.array(sample_sizes[:len(comb_mae)]),
            np.array(comb_mae),
            color='purple', label="Ensemble")

    # x_ticks_labels_list = list(np.array(["1%", "2%", "3%", "4%", "5%", "6%", "7%", "8%", "9%", "10%", "13%", "17%", "21%", "25%", "28%", "34%", "40%", "43%", "47%", "50%"]))
    # ax.set_xticks(np.append(x, pred_sample_size))

    # if dataset == "AG_NEWS":
    #     x_ticks = np.array([840, 1680, 2520, 3360, 4200, 5040, 5880, 6720, 7560, 8400, 12600, 16800, 21000, 25200, 29400, 33600, 37800, 42000])
    # elif dataset == "IMDB":
    #     x_ticks = np.array([175, 350, 525, 700, 875, 1050, 1225, 1400, 1575, 1750, 2625, 3500, 4375, 5250, 6125, 7000, 7875, 8750])
    #
    # ax.set_xticks(x_ticks)
    # x_ticks_labels_list = list(np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "25", "30", "35", "40", "45", "50"]))

    if dataset == "AG_NEWS":
        x_ticks = np.array([840, 8400, 16800, 25200, 33600, 42000])
    elif dataset == "IMDB":
        x_ticks = np.array([175, 1750, 3500, 5250, 7000, 8750])
    elif dataset == "SST2":
        x_ticks = np.array([471, 4714, 9428, 14143, 18857, 23572])
    elif dataset == "DBpedia":
        x_ticks = np.array([3920, 39200, 78400, 117600, 156800, 196000])

    ax.set_xticks(x_ticks)
    x_ticks_labels_list = list(np.array(["1", "10", "20", "30", "40", "50"]))

    ax.set_ylabel("Mean Absolute Error", fontsize=20)
    ax.yaxis.set_tick_params(labelsize=18)
    # ax.set_xticks(sample_sizes[:len(exp_mae)])
    if show_percentage_labels:
        ax.set_xticklabels(x_ticks_labels_list, rotation=0, fontsize=18)
        ax.set_xlabel("Sample size (%)", fontsize=20)
    else:
        # ax.set_xticklabels(sample_sizes[:len(exp_mae)], rotation=90, fontsize=8)
        ax.set_xlabel("Sample size", fontsize=20)
    # ax.yaxis.set_tick_params(labelsize=10)
    if sampling_method == "random":
        ax.set_title(f"{dataset} (Random sampling)", fontsize=22)
    else:
        ax.set_title(f"{dataset} (Perplexity sampling)", fontsize=22)
    ax.legend(loc=(0.75, 0.65), fontsize=18)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    if sampling_method == "random":
        plt.savefig(f'plots/{dataset}_RS.png')
    else:
        plt.savefig(f'plots/{dataset}_PPL.png')
    # plt.savefig(f'plots/sampleSize_vs_MAE_{dataset}.png')


def fit_and_predict_combined_extrapolating_functions(train_acc, sample_sizes, pred_sample_size, training_index, show_percentage_labels, sampling_method):
    testing_sample_sizes = sample_sizes[20:len(sample_sizes)]
    testing_accuracies = train_acc[20:len(train_acc)]
    testing_mean_acc = [np.mean(i) for i in testing_accuracies]

    print(f"testing sample sizes: {testing_sample_sizes}")
    print(f"testing accuracies: {testing_accuracies}")
    print(f"testing mean accuracies: {testing_mean_acc}")

    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    slope, intercept = np.polyfit(np.array(sample_sizes), np.array(mean_acc), 1)

    training_indexes = list(range(0, training_index))
    # training_indexes = [4, 9, 11, 13, 15, 17, 19]

    training_sample_sizes = [sample_sizes[i] for i in training_indexes]
    training_accuracies = [train_acc[i] for i in training_indexes]
    training_mean_acc = [np.mean(i) for i in training_accuracies]

    error_training = [error[i] for i in training_indexes]

    training_sample_sizes_array = np.array(training_sample_sizes)
    training_mean_acc_array = np.array(training_mean_acc)

    print(f"training_sample_sizes_array: {training_sample_sizes_array}")
    print(f"training_mean_acc_array: {training_mean_acc_array}")

    def best_fit_line(x, a, b):
        return a * x + b

    def exp_func(x, a, b):
        return a * (x ** b)

    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    def power4_func(x, a, b, c, d):
        return a - (((b * x) + c) ** (-d))

    def exp_inv_power4_combined_func(x, a, b, c, d, e, f, g, h, i):
        return (a * (x ** b)) + ((1 - c) - d * (x ** e)) + (f - (((g * x) + h) ** (-i)))

    def custom_loss_function(y_pred, y_true, sigma):
        squared_difference = ((y_pred - y_true) ** 2) * sigma
        loss = torch.mean(squared_difference)
        return loss

    sigma = np.ones(len(training_sample_sizes))

    start_index = len(sigma)
    stop_index = 0
    step = -1
    count = 0

    # applying for loop
    for i in range(start_index, stop_index, step):
        sigma[count] = i / len(sigma)
        count = count + 1

    popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array, maxfev=1000000,
                        sigma=sigma, bounds=(-7, 7))
    a, b = popt

    popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array, maxfev=1000000,
                        sigma=sigma, bounds=(-7, 7))
    c, d, e = popt

    if dataset == "SST2":
        popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                            sigma=sigma, maxfev=10000000)
        f, g, h, i = popt
    else:
        popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                            sigma=sigma, maxfev=1000000)
        f, g, h, i = popt

    # Define variables for learning rate and number of epochs for fitting with TF
    if dataset == "IMDB":
        learning_rate = 0.01
    elif dataset == "SST2" or dataset == "AG_NEWS" or dataset == "DBpedia":
        learning_rate = 0.001

    sigma_ensemble = np.ones(len(training_sample_sizes))
    start_index = 0
    stop_index = len(sigma)
    step = 1

    # applying for loop
    for index in range(start_index, stop_index, step):
        sigma_ensemble[index] = (index + 1) / len(sigma_ensemble)

    training_epochs = 5000
    mse = nn.MSELoss()

    j = torch.tensor(0.0, requires_grad=True)
    k = torch.tensor(0.0, requires_grad=True)
    l = torch.tensor(0.0, requires_grad=True)
    m = torch.tensor(0.0, requires_grad=True)
    n = torch.tensor(0.0, requires_grad=True)
    o = torch.tensor(0.0, requires_grad=True)
    p = torch.tensor(0.0, requires_grad=True)
    q = torch.tensor(0.0, requires_grad=True)
    r = torch.tensor(0.0, requires_grad=True)

    for epoch in range(training_epochs):
        y_pred = exp_inv_power4_combined_func(torch.tensor(training_sample_sizes, dtype=torch.float32), j, k, l, m,
                                              n, o, p, q, r)
        cost_function = mse(y_pred, torch.tensor(training_mean_acc, dtype=torch.float32))
        # cost_function = custom_loss_function(y_pred, torch.tensor(training_mean_acc, dtype=torch.float32), torch.tensor(sigma_ensemble, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            j -= j.grad * learning_rate
            k -= k.grad * learning_rate
            l -= l.grad * learning_rate
            m -= m.grad * learning_rate
            n -= n.grad * learning_rate
            o -= o.grad * learning_rate
            p -= p.grad * learning_rate
            q -= q.grad * learning_rate
            r -= r.grad * learning_rate
            j.grad.zero_()
            k.grad.zero_()
            l.grad.zero_()
            m.grad.zero_()
            n.grad.zero_()
            o.grad.zero_()
            p.grad.zero_()
            q.grad.zero_()
            r.grad.zero_()

    # print(f"Curve fit weights exp func with weighted nls: a = {a} and b = {b}.")
    # print(f"Curve fit weights inverse power law func with weighted nls: a = {c}, b = {d} and c = {e}.")
    # print(f"Curve fit weights power4 func with weighted nls: a = {f}, b = {g}, c = {h} and d = {i}.")
    # print(f"Curve fit weights ensemble func with unweighted gd: a = {j}, b = {k}, c = {l}, d = {m}, e = {n}, f = {o}, g= {p}, h = {q} and i = {r}.")

    max_acc_exp = exp_func(
        torch.tensor(pred_sample_size, dtype=torch.float32), a, b).detach().numpy()

    max_acc_inv = inverse_power_law_func(
        torch.tensor(pred_sample_size, dtype=torch.float32), c, d, e).detach().numpy()

    max_acc_pow4 = power4_func(
        torch.tensor(pred_sample_size, dtype=torch.float32), f, g, h, i).detach().numpy()

    max_acc_comb = exp_inv_power4_combined_func(
        torch.tensor(pred_sample_size, dtype=torch.float32), j, k, l, m, n, o, p, q, r).detach().numpy()

    # print(
    #     f"A model accuracy of {max_acc_exp} is predicted for {pred_sample_size} samples "
    #     f"using exp func with nls weighted.")
    #
    # print(
    #     f"A model accuracy of {max_acc_inv} is predicted for {pred_sample_size} samples "
    #     f"using inverse power law func with nls weighted.")
    #
    # print(
    #     f"A model accuracy of {max_acc_pow4} is predicted for {pred_sample_size} samples "
    #     f"using power4 func with nls weighted.")
    #
    print(
        f"A model accuracy of {max_acc_comb} is predicted for {pred_sample_size} samples "
        f"using ensemble func with gd unweighted.")

    mae = nn.L1Loss()

    output_exp = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), a, b),
                     torch.tensor(testing_mean_acc, dtype=torch.float32))
    output_inv = mae(inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), c, d, e),
                     torch.tensor(testing_mean_acc, dtype=torch.float32))
    output_pow4 = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), f, g, h, i),
                      torch.tensor(testing_mean_acc, dtype=torch.float32))
    output_comb = mae(exp_inv_power4_combined_func(torch.tensor(testing_sample_sizes, dtype=torch.float32),
                                                   j, k, l, m, n, o, p, q, r),
                      torch.tensor(testing_mean_acc, dtype=torch.float32))

    print(f"The mae for the exp func curve fit is {output_exp}.")
    print(f"The mae for the inverse power law func curve fit is {output_inv}.")
    print(f"The mae for the power4 func curve fit is {output_pow4}.")
    print(f"The mae for the comb func curve fit is {output_comb}.")

    # x_cont_training = np.linspace(x[0], (pred_sample_size / 2), 50)
    # x_cont_testing = np.linspace(((pred_sample_size / 2) + 50), pred_sample_size, 50)
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if training_index < 20:
        ax.errorbar(x[0:20], mean_acc[0:20], yerr=error[0:20], fmt="o", color="teal")
    ax.errorbar(testing_sample_sizes, testing_mean_acc, yerr=error[20:len(x)], fmt="o", color="darkorange",
                label="Testing Sample")
    ax.errorbar(training_sample_sizes, training_mean_acc, yerr=error_training, fmt="o", color="midnightblue",
                label="Training Sample")

    ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), a, b).detach().numpy(),
            color='red', label="Exp")
    # ax.plot(x_cont_testing, exp_func(torch.tensor(x_cont_testing, dtype=torch.float32), a, b).detach().numpy(),
    #         color='red', linestyle='dashed')

    ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), c, d, e).detach().numpy(),
            color='blue', label="Inv")
    # ax.plot(x_cont_testing,
    #         inverse_power_law_func(torch.tensor(x_cont_testing, dtype=torch.float32), c, d, e).detach().numpy(),
    #         color='blue', linestyle='dashed')

    ax.plot(x_cont,
            power4_func(torch.tensor(x_cont, dtype=torch.float32), f, g, h, i).detach().numpy(),
            color='green', label="Pow4")
    # ax.plot(x_cont_testing, power4_func(torch.tensor(x_cont_testing, dtype=torch.float32), f, g, h, i).detach().numpy(),
    #         color='green', linestyle='dashed')

    ax.plot(x_cont, exp_inv_power4_combined_func(torch.tensor(x_cont, dtype=torch.float32), j, k, l, m, n, o, p, q,
                                                 r).detach().numpy(),
            color="purple", label="Ensemble")

    # ax.plot(x_cont,
    #         best_fit_line(torch.tensor(x_cont, dtype=torch.float32), slope, intercept).detach().numpy(),
    #         color='teal', label="Best fit line")

    ax.set_ylabel("Performance (acc.)", fontsize=12)

    # if dataset == "AG_NEWS":
    #     x_ticks = np.array([840, 1680, 2520, 3360, 4200, 5040, 5880, 6720, 7560, 8400, 16800, 25200, 33600, 42000, 50400, 58800, 67200, 75600, 84000])
    # elif dataset == "IMDB":
    #     x_ticks = np.array([175, 350, 525, 700, 875, 1050, 1225, 1400, 1575, 1750, 3500, 5250, 7000, 8750, 10500, 12250, 14000, 15750, 17500])
    #
    # ax.set_xticks(x_ticks)
    # # ax.set_xticks(np.append(x, pred_sample_size))
    # x_ticks_labels_list = list(np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]))

    if dataset == "AG_NEWS":
        x_ticks = np.array([840, 8400, 16800, 25200, 33600, 42000, 50400, 58800, 67200, 75600, 84000])
    elif dataset == "IMDB":
        x_ticks = np.array([175, 1750, 3500, 5250, 7000, 8750, 10500, 12250, 14000, 15750, 17500])
    elif dataset == "SST2":
        x_ticks = np.array([471, 4714, 9428, 14143, 18857, 23572, 28286, 33000, 37715, 42429, 47144])
    elif dataset == "DBpedia":
        x_ticks = np.array([3920, 39200, 78400, 117600, 156800, 196000, 235200, 274400, 313600, 352800, 392000])

    ax.set_xticks(x_ticks)
    # ax.set_xticks(np.append(x, pred_sample_size))
    x_ticks_labels_list = list(np.array(["1", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]))

    # yticks = np.append(mean_acc, max_acc_exp)
    # yticks = np.append(yticks, max_acc_inv)
    # yticks = np.append(yticks, max_acc_pow4)
    # ax.set_yticks(np.append(yticks, max_acc_comb))
    # ytickLabelsList = list(np.round(np.append(yticks, max_acc_comb), 4))
    # ax.set_yticklabels(ytickLabelsList, fontsize=5)

    if dataset == "AG_NEWS" and training_index < 20:
        if sampling_method == "random":
            y_ticks = [0.50, 0.70, 0.90, 1.1, 1.3]
        else:
            y_ticks = [0.50, 0.60, 0.70, 0.80, 0.90]
    elif dataset == "AG_NEWS" and training_index >= 20:
        if sampling_method == "random":
            y_ticks = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        else:
            y_ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    if dataset == "IMDB" and training_index < 20:
        y_ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    elif dataset == "IMDB" and training_index >= 20:
        y_ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    if dataset == "SST2" and training_index < 20:
        y_ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    elif dataset == "SST2" and training_index >= 20:
        y_ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    if dataset == "DBpedia" and training_index < 20:
        y_ticks = [0.40, 0.60, 0.80, 1.0, 1.2]
    elif dataset == "DBpedia" and training_index >= 20:
        y_ticks = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    ax.set_yticks(y_ticks)

    if show_percentage_labels:
        ax.set_xticklabels(x_ticks_labels_list, rotation=0, fontsize=8)
        ax.set_xlabel("Sample size (%)", fontsize=12)
    else:
        ax.set_xlabel("Sample size", fontsize=12)

    ax.yaxis.set_tick_params(labelsize=10)

    v_line = x_ticks[10]/2
    ax.axvline(x=v_line, color="grey", linestyle='--')

    if sampling_method == "random":
        ax.set_title(f"{dataset} (Random sampling)", fontsize=14)
    else:
        ax.set_title(f"{dataset} (Perplexity sampling)", fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    # if dataset == "IMDB":
    #     ax.text(6300, 0.52, f'MAE for Exp func: {output_exp}', color='red')
    #     ax.text(6300, 0.5, f'MAE for Inv func: {output_inv}', color='blue')
    #     ax.text(6300, 0.48, f'MAE for Pow4 func: {output_pow4}', color='green')
    #     ax.text(6300, 0.46, f'MAE for Ensemble func: {output_comb}', color='purple')
    # elif dataset == "SST2":
    #     ax.text(6500, 0.60, f'MAE for Exp func: {output_exp}', color='red')
    #     ax.text(6500, 0.58, f'MAE for Inv func: {output_inv}', color='blue')
    #     ax.text(6500, 0.56, f'MAE for Pow4 func: {output_pow4}', color='green')
    #     ax.text(6500, 0.54, f'MAE for Ensemble func: {output_comb}', color='purple')
    # elif dataset == "AG_NEWS":
    #     if training_index == 20:
    #         ax.text(28560, 0.47, f'MAE for Exp func: {output_exp}', color='red')
    #         ax.text(28560, 0.45, f'MAE for Inv func: {output_inv}', color='blue')
    #         ax.text(28560, 0.43, f'MAE for Pow4 func: {output_pow4}', color='green')
    #         ax.text(28560, 0.41, f'MAE for Ensemble func: {output_comb}', color='purple')
    #     else:
    #         ax.text(28560, 0.53, f'MAE for Exp func: {output_exp}', color='red')
    #         ax.text(28560, 0.49, f'MAE for Inv func: {output_inv}', color='blue')
    #         ax.text(28560, 0.45, f'MAE for Pow4 func: {output_pow4}', color='green')
    #         ax.text(28560, 0.41, f'MAE for Ensemble func: {output_comb}', color='purple')
    # elif dataset == "DBpedia":
    #     ax.text(140000, 0.47, f'MAE for Exp func: {output_exp}', color='red')
    #     ax.text(140000, 0.43, f'MAE for Inv func: {output_inv}', color='blue')
    #     ax.text(140000, 0.39, f'MAE for Pow4 func: {output_pow4}', color='green')
    #     ax.text(140000, 0.35, f'MAE for Ensemble func: {output_comb}', color='purple')

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    if sampling_method == "random":
        if training_index < 20:
            plt.savefig(f'plots/{dataset}_RS_10%.png')
        else:
            plt.savefig(f'plots/{dataset}_RS_50%.png')
    else:
        if training_index < 20:
            plt.savefig(f'plots/{dataset}_PPL_10%.png')
        else:
            plt.savefig(f'plots/{dataset}_PPL_50%.png')

    # plt.savefig(f'plots/learning_curves_all_extrapolating_functions_{dataset}.png')

    # with open("results/demo_extrapolating_func_nls_unweighted_" + str(n_epochs) + 'epochs' + '.txt', 'a+') as file:
    #     file.write(f'Exp func using unweighted nls:\n')
    #     file.write(f'Curve fit weights: a = {a} and b = {b}.\n')
    #     file.write(
    #         f'A model accuracy of {max_acc_exp} is predicted for {pred_sample_size} samples.\n')
    #     file.write(f'The mae for the exp curve fit is {output_exp}.\n\n')
    #
    #     file.write(f'Inverse power law func using unweighted nls:\n')
    #     file.write(f'Curve fit weights: a = {c}, b = {d} and c = {e}.\n')
    #     file.write(
    #         f'A model accuracy of {max_acc_inv} is predicted for {pred_sample_size} samples.\n')
    #     file.write(f'The mae for the inverse power law curve fit is {output_inv}.\n\n')
    #
    #     file.write(f'Power4 func using unweighted nls:\n')
    #     file.write(f'Curve fit weights: a = {f}, b = {g}, c = {h} and d = {i}.\n')
    #     file.write(
    #         f'A model accuracy of {max_acc_pow4} is predicted for {pred_sample_size} samples.\n')
    #     file.write(f'The mae for the power4 curve fit is {output_pow4}.\n\n')


def fit_and_predict_combined_parameters_minimization_methods(train_acc, sample_sizes, pred_sample_size, n_epochs,
                                                             function_name, show_percentage_labels, random_init,
                                                             seed_option):
    if seed_option == 1:
        random_seed = 42
    else:
        random_seed = 39

    testing_sample_sizes = sample_sizes[20:len(sample_sizes)]
    testing_accuracies = train_acc[20:len(train_acc)]
    testing_mean_acc = [np.mean(i) for i in testing_accuracies]

    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    training_indexes = [4, 9, 13, 19]
    training_sample_sizes = [sample_sizes[i] for i in training_indexes]
    training_accuracies = [train_acc[i] for i in training_indexes]
    training_mean_acc = [np.mean(i) for i in training_accuracies]

    error_training = [error[i] for i in training_indexes]

    training_sample_sizes_array = np.array(training_sample_sizes)
    training_mean_acc_array = np.array(training_mean_acc)

    # Define mean squared error cost and exponential curve fit functions
    mse = nn.MSELoss()

    def exp_func(x, a, b):
        return a * (x ** b)

    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    def power4_func(x, a, b, c, d):
        return a - (((b * x) + c) ** (-d))

    def custom_loss_function(y_pred, y_true, sigma):
        squared_difference = ((y_pred - y_true) ** 2) * sigma
        loss = torch.mean(squared_difference)
        return loss

    # Define variables for learning rate and number of epochs for fitting with TF
    learning_rate = 0.01
    training_epochs = 5000

    if function_name == "inv":
        # Define variables for inverse power law gd unweighted, learning rate and number of epochs for fitting with TF
        if random_init:
            torch.manual_seed(random_seed)
            j = torch.randn(1, requires_grad=True)
            k = torch.randn(1, requires_grad=True)
            l = torch.randn(1, requires_grad=True)
        else:
            j = torch.tensor(0.0, requires_grad=True)
            k = torch.tensor(0.0, requires_grad=True)
            l = torch.tensor(0.0, requires_grad=True)
    elif function_name == "exp":
        if random_init:
            torch.manual_seed(random_seed)
            j = torch.randn(1, requires_grad=True)
            k = torch.randn(1, requires_grad=True)
        else:
            j = torch.tensor(0.0, requires_grad=True)
            k = torch.tensor(0.0, requires_grad=True)
    elif function_name == "pow4":
        if random_init:
            torch.manual_seed(random_seed)
            j = torch.randn(1, requires_grad=True)
            k = torch.randn(1, requires_grad=True)
            l = torch.randn(1, requires_grad=True)
            j1 = torch.randn(1, requires_grad=True)
        else:
            j = torch.tensor(0.0, requires_grad=True)
            k = torch.tensor(0.0, requires_grad=True)
            l = torch.tensor(0.0, requires_grad=True)
            j1 = torch.tensor(0.0, requires_grad=True)

    # Fit the function to the data
    # gd unweighted
    for epoch in range(training_epochs):
        if function_name == "inv":
            y_pred = inverse_power_law_func(torch.tensor(training_sample_sizes, dtype=torch.float32), j, k, l)
        elif function_name == "exp":
            y_pred = exp_func(torch.tensor(training_sample_sizes, dtype=torch.float32), j, k)
        elif function_name == "pow4":
            y_pred = power4_func(torch.tensor(training_sample_sizes, dtype=torch.float32), j, k, l, j1)

        cost_function = mse(y_pred, torch.tensor(training_mean_acc, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            if function_name == "inv":
                j -= j.grad * learning_rate
                k -= k.grad * learning_rate
                l -= l.grad * learning_rate
                j.grad.zero_()
                k.grad.zero_()
                l.grad.zero_()
            elif function_name == "exp":
                j -= j.grad * learning_rate
                k -= k.grad * learning_rate
                j.grad.zero_()
                k.grad.zero_()
            elif function_name == "pow4":
                j -= j.grad * learning_rate
                k -= k.grad * learning_rate
                l -= l.grad * learning_rate
                j1 -= j1.grad * learning_rate
                j.grad.zero_()
                k.grad.zero_()
                l.grad.zero_()
                j1.grad.zero_()

    sample_sizes_array = np.array(sample_sizes)
    mean_acc_array = np.array(mean_acc)
    sigma = np.ones(len(training_sample_sizes))

    start_index = len(sigma)
    stop_index = 0
    step = -1
    count = 0

    # applying for loop
    for i in range(start_index, stop_index, step):
        sigma[count] = i / len(sigma)
        count = count + 1

    # nls weighted
    if function_name == "inv":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random(), random.random()]
            popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array,
                                sigma=sigma, maxfev=1000, bounds=(-7, 7), p0=init_vals)
            c, d, e = popt
        else:
            popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array,
                                sigma=sigma, maxfev=1000, bounds=(-7, 7))
            c, d, e = popt

    elif function_name == "exp":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random()]
            popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array,
                                sigma=sigma, maxfev=1000, bounds=(-7, 7), p0=init_vals)
            c, d = popt
        else:
            popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array,
                                sigma=sigma, maxfev=1000, bounds=(-7, 7))
            c, d = popt

    elif function_name == "pow4":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random(), random.random(), random.random()]
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                sigma=sigma, maxfev=1000000, p0=init_vals)
            c, d, e, c1 = popt
        else:
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                sigma=sigma, maxfev=1000000)
            c, d, e, c1 = popt

    # nls unweighted
    if function_name == "inv":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random(), random.random()]
            popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array, maxfev=1000,
                                bounds=(-7, 7), p0=init_vals)
            f, g, h = popt
        else:
            popt, _ = curve_fit(inverse_power_law_func, training_sample_sizes_array, training_mean_acc_array,
                                maxfev=1000,
                                bounds=(-7, 7))
            f, g, h = popt

    elif function_name == "exp":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random()]
            popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array, maxfev=1000, bounds=(-7, 7),
                                p0=init_vals)
            f, g = popt
        else:
            popt, _ = curve_fit(exp_func, training_sample_sizes_array, training_mean_acc_array, maxfev=1000,
                                bounds=(-7, 7))
            f, g = popt

    elif function_name == "pow4":
        if random_init:
            random.seed(random_seed)
            init_vals = [random.random(), random.random(), random.random(), random.random()]
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                maxfev=1000000, p0=init_vals)
            f, g, h, f1 = popt
        else:
            popt, _ = curve_fit(power4_func, training_sample_sizes_array, training_mean_acc_array, method='dogbox',
                                maxfev=1000000)
            f, g, h, f1 = popt


    # Define variables for weighted gd
    if function_name == "inv":
        if random_init:
            torch.manual_seed(random_seed)
            m = torch.randn(1, requires_grad=True)
            n = torch.randn(1, requires_grad=True)
            o = torch.randn(1, requires_grad=True)
        else:
            m = torch.tensor(0.0, requires_grad=True)
            n = torch.tensor(0.0, requires_grad=True)
            o = torch.tensor(0.0, requires_grad=True)

    elif function_name == "exp":
        if random_init:
            torch.manual_seed(random_seed)
            m = torch.randn(1, requires_grad=True)
            n = torch.randn(1, requires_grad=True)
        else:
            m = torch.tensor(0.0, requires_grad=True)
            n = torch.tensor(0.0, requires_grad=True)

    elif function_name == "pow4":
        if random_init:
            torch.manual_seed(random_seed)
            m = torch.randn(1, requires_grad=True)
            n = torch.randn(1, requires_grad=True)
            o = torch.randn(1, requires_grad=True)
            m1 = torch.randn(1, requires_grad=True)
        else:
            m = torch.tensor(0.0, requires_grad=True)
            n = torch.tensor(0.0, requires_grad=True)
            o = torch.tensor(0.0, requires_grad=True)
            m1 = torch.tensor(0.0, requires_grad=True)

    sigma = np.ones(len(training_sample_sizes))
    start_index = 0
    stop_index = len(sigma)
    step = 1

    # applying for loop
    for i in range(start_index, stop_index, step):
        sigma[i] = (i + 1) / len(sigma)

    # training_sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # sigma = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Fit the inverse power law function to the data
    # gd weighted
    for epoch in range(training_epochs):
        if function_name == "inv":
            y_pred = inverse_power_law_func(torch.tensor(training_sample_sizes, dtype=torch.float32), m, n, o)

        elif function_name == "exp":
            y_pred = exp_func(torch.tensor(training_sample_sizes, dtype=torch.float32), m, n)

        elif function_name == "pow4":
            y_pred = power4_func(torch.tensor(training_sample_sizes, dtype=torch.float32), m, n, o, m1)

        cost_function = custom_loss_function(y_pred,
                                             torch.tensor(training_mean_acc, dtype=torch.float32),
                                             torch.tensor(sigma, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            if function_name == "inv":
                m -= m.grad * learning_rate
                n -= n.grad * learning_rate
                o -= o.grad * learning_rate
                m.grad.zero_()
                n.grad.zero_()
                o.grad.zero_()

            elif function_name == "exp":
                m -= m.grad * learning_rate
                n -= n.grad * learning_rate
                m.grad.zero_()
                n.grad.zero_()

            elif function_name == "pow4":
                m -= m.grad * learning_rate
                n -= n.grad * learning_rate
                o -= o.grad * learning_rate
                m1 -= m1.grad * learning_rate
                m.grad.zero_()
                n.grad.zero_()
                o.grad.zero_()
                m1.grad.zero_()

    if function_name == "inv":
        print(f"Curve fit weights inverse power law func with weighted nls: c = {c}, d = {d} and e = {e}.")
        print(f"Curve fit weights inverse power law func with unweighted nls: f = {f}, g = {g} and h = {h}.")
        print(f"Curve fit weights inverse power law func with unweighted gd: j = {j}, k = {k} and l = {l}.")
        print(f"Curve fit weights inverse power law func with weighted gd: m = {m}, n = {n} and o = {o}.")

    elif function_name == "exp":
        print(f"Curve fit weights exp func with weighted nls: c = {c} and d = {d}.")
        print(f"Curve fit weights exp func with unweighted nls: f = {f} and g = {g}.")
        print(f"Curve fit weights exp func with unweighted gd: j = {j} and k = {k}.")
        print(f"Curve fit weights exp func with weighted gd: m = {m} and n = {n}.")

    elif function_name == "pow4":
        print(f"Curve fit weights power4 func with weighted nls: c = {c}, d = {d}, e = {e} and c1 = {c1}.")
        print(f"Curve fit weights power4 func with unweighted nls: f = {f}, g = {g}, h = {h} and f1 = {f1}.")
        print(f"Curve fit weights power4 func with unweighted gd: j = {j}, k = {k}, l = {l} and j1 = {j1}.")
        print(f"Curve fit weights power4 func with weighted gd: m = {m}, n = {n}, o = {o} and m1 = {m1}.")

    # We can now estimate the accuracy for pred_sample_size
    if function_name == "inv":
        max_acc_nls_weighted = inverse_power_law_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), c, d,
            e).detach().numpy()
        max_acc_nls_unweighted = inverse_power_law_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), f, g,
            h).detach().numpy()
        max_acc_gd_unweighted = inverse_power_law_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), j, k,
            l).detach().numpy()
        max_acc_gd_weighted = inverse_power_law_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), m, n,
            o).detach().numpy()

    elif function_name == "exp":
        max_acc_nls_weighted = exp_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), c, d, ).detach().numpy()
        max_acc_nls_unweighted = exp_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), f, g, ).detach().numpy()
        max_acc_gd_unweighted = exp_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), j, k, ).detach().numpy()
        max_acc_gd_weighted = exp_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), m, n, ).detach().numpy()

    elif function_name == "pow4":
        max_acc_nls_weighted = power4_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), c, d,
            e, c1).detach().numpy()
        max_acc_nls_unweighted = power4_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), f, g,
            h, f1).detach().numpy()
        max_acc_gd_unweighted = power4_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), j, k,
            l, j1).detach().numpy()
        max_acc_gd_weighted = power4_func(
            torch.tensor(pred_sample_size, dtype=torch.float32), m, n,
            o, m1).detach().numpy()
    # Print predicted x value and append to plot values
    print(
        f"A model accuracy of {max_acc_nls_weighted} is predicted for {pred_sample_size} samples "
        f"using {function_name} func with nls weighted.")
    print(
        f"A model accuracy of {max_acc_nls_unweighted} is predicted for {pred_sample_size} samples "
        f"using {function_name} func with nls unweighted.")
    print(
        f"A model accuracy of {max_acc_gd_unweighted} is predicted for {pred_sample_size} samples "
        f"using {function_name} func with gd unweighted.")
    print(
        f"A model accuracy of {max_acc_gd_weighted} is predicted for {pred_sample_size} samples "
        f"using {function_name} func with gd weighted.")

    mae = nn.L1Loss()
    if function_name == "inv":
        output_nls_weighted = mae(
            inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), c, d, e),
            torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_nls_unweighted = mae(
            inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), f, g, h),
            torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_unweighted = mae(
            inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), j, k, l),
            torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_weighted = mae(
            inverse_power_law_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), m, n, o),
            torch.tensor(testing_mean_acc, dtype=torch.float32))

    elif function_name == "exp":
        output_nls_weighted = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), c, d),
                                  torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_nls_unweighted = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), f, g),
                                    torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_unweighted = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), j, k),
                                   torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_weighted = mae(exp_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), m, n),
                                 torch.tensor(testing_mean_acc, dtype=torch.float32))

    elif function_name == "pow4":
        output_nls_weighted = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), c, d, e, c1),
                                  torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_nls_unweighted = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), f, g, h, f1),
                                    torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_unweighted = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), j, k, l, j1),
                                   torch.tensor(testing_mean_acc, dtype=torch.float32))
        output_gd_weighted = mae(power4_func(torch.tensor(testing_sample_sizes, dtype=torch.float32), m, n, o, m1),
                                 torch.tensor(testing_mean_acc, dtype=torch.float32))

    print(f"The mae for the {function_name} curve fit is {output_nls_weighted} using weighted nls.")
    print(f"The mae for the {function_name} curve fit is {output_nls_unweighted} using unweighted nls.")
    print(f"The mae for the {function_name} curve fit is {output_gd_unweighted} using unweighted gd.")
    print(f"The mae for the {function_name} curve fit is {output_gd_weighted} using weighted gd.")

    x_cont = np.linspace(x[0], pred_sample_size, 100)
    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(training_sample_sizes, training_mean_acc, yerr=error_training, fmt="o", label="Training Sample")
    if function_name == "inv":
        ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), c, d, e).detach().numpy(),
                color='green', label="Weighted non-linear least squares")
        ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), f, g, h).detach().numpy(),
                color='blue', label="Unweighted non-linear least squares")
        ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), j, k, l).detach().numpy(),
                color='orange', label="Unweighted gradient descent")
        ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), m, n, o).detach().numpy(),
                color='red', label="Weighted gradient descent")

    elif function_name == "exp":
        ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), c, d).detach().numpy(),
                color='green', label="Weighted non-linear least squares")
        ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), f, g).detach().numpy(),
                color='blue', label="Unweighted non-linear least squares")
        ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), j, k).detach().numpy(),
                color='orange', label="Unweighted gradient descent")
        ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), m, n).detach().numpy(),
                color='red', label="Weighted gradient descent")

    elif function_name == "pow4":
        ax.plot(x_cont, power4_func(torch.tensor(x_cont, dtype=torch.float32), c, d, e, c1).detach().numpy(),
                color='green', label="Weighted non-linear least squares")
        ax.plot(x_cont, power4_func(torch.tensor(x_cont, dtype=torch.float32), f, g, h, f1).detach().numpy(),
                color='blue', label="Unweighted non-linear least squares")
        ax.plot(x_cont, power4_func(torch.tensor(x_cont, dtype=torch.float32), j, k, l, j1).detach().numpy(),
                color='orange', label="Unweighted gradient descent")
        ax.plot(x_cont, power4_func(torch.tensor(x_cont, dtype=torch.float32), m, n, o, m1).detach().numpy(),
                color='red', label="Weighted gradient descent")

    ax.set_ylabel("Performance (acc.)", fontsize=12)
    ax.set_xlabel("Sample size (%)", fontsize=12)
    ax.set_xticks(np.append(training_sample_sizes, pred_sample_size))

    # yticks = np.append(training_mean_acc, max_acc_nls_weighted)
    # yticks = np.append(yticks, max_acc_nls_unweighted)
    # yticks = np.append(yticks, max_acc_gd_unweighted)
    # ax.set_yticks(np.append(yticks, max_acc_gd_weighted))

    y_ticks = [0.50, 0.60, 0.70, 0.80]
    ax.set_yticks(y_ticks)
    
    x_ticks_labels_list = list(np.array(["5", "10", "25", "50", "100"]))

    if show_percentage_labels:
        ax.set_xticklabels(x_ticks_labels_list, rotation=0, fontsize=10)
    else:
        ax.set_xticklabels(list(np.append(training_mean_acc, pred_sample_size)), rotation=90, fontsize=10)

    ax.axvline(x=8750, color="grey", linestyle='--')

    ax.yaxis.set_tick_params(labelsize=10)
    if function_name == "inv":
        ax.set_title(f"Inv function - {dataset}", fontsize=14)
    elif function_name == "exp":
        ax.set_title(f"Exp function - {dataset}", fontsize=14)
    elif function_name == "pow4":
        ax.set_title(f"Pow4 function - {dataset}", fontsize=14)
    # if function_name == "inv":
    #     ax.text(6500, 0.34, f'MAE for nls weighted: {output_nls_weighted}', color='green')
    #     ax.text(6500, 0.32, f'MAE for nls unweighted: {output_nls_unweighted}', color='blue')
    #     ax.text(6500, 0.30, f'MAE for gd unweighted: {output_gd_unweighted}', color='orange')
    #     ax.text(6500, 0.28, f'MAE for gd weighted: {output_gd_weighted}', color='red')
    # else:
    #     ax.text(6500, 0.54, f'MAE for nls weighted: {output_nls_weighted}', color='green')
    #     ax.text(6500, 0.52, f'MAE for nls unweighted: {output_nls_unweighted}', color='blue')
    #     ax.text(6500, 0.50, f'MAE for gd unweighted: {output_gd_unweighted}', color='orange')
    #     ax.text(6500, 0.48, f'MAE for gd weighted: {output_gd_weighted}', color='red')
    ax.legend(loc='lower right', fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/minimization_methods_combined_{function_name}.png')

    with open(output_results_filename + str(n_epochs) + 'epochs_' + function_name + '.txt', 'a+') as file:
        file.write(f'{function_name} func using unweighted nls:\n')
        if function_name == "inv":
            file.write(f'Curve fit weights: a = {f}, b = {g} and c = {h}.\n')
        elif function_name == "exp":
            file.write(f'Curve fit weights: a = {f} and b = {g}.\n')
        elif function_name == "pow4":
            file.write(f'Curve fit weights: a = {f}, b = {g}, c = {h} and d = {f1}.\n')
        file.write(
            f'A model accuracy of {max_acc_nls_unweighted} is predicted for {pred_sample_size} samples.\n')
        file.write(f'The mae for the curve fit is {output_nls_unweighted}.\n\n')

        file.write(f'{function_name} func using weighted nls:\n')
        if function_name == "inv":
            file.write(f'Curve fit weights: a = {c}, b = {d} and c = {e}.\n')
        elif function_name == "exp":
            file.write(f'Curve fit weights: a = {c} and b = {d}.\n')
        elif function_name == "pow4":
            file.write(f'Curve fit weights: a = {c}, b = {d}, c = {e} and d = {c1}.\n')
        file.write(
            f'A model accuracy of {max_acc_nls_weighted} is predicted for {pred_sample_size} samples.\n')
        file.write(f'The mae for the curve fit is {output_nls_weighted}.\n\n')

        file.write(f'{function_name} func using unweighted gd:\n')

        if function_name == "inv":
            file.write(f'Curve fit weights: a = {j}, b = {k} and c = {l}.\n')
        elif function_name == "exp":
            file.write(f'Curve fit weights: a = {j} and b = {k}.\n')
        elif function_name == "pow4":
            file.write(f'Curve fit weights: a = {j}, b = {k}, c = {l} and d = {j1}.\n')
        file.write(
            f'A model accuracy of {max_acc_gd_unweighted} is predicted for {pred_sample_size} samples.\n')
        file.write(f'The mae for the curve fit is {output_gd_unweighted}.\n\n')

        file.write(f'{function_name} func using weighted gd:\n')
        if function_name == "inv":
            file.write(f'Curve fit weights: a = {m}, b = {n} and c = {o}.\n')
        elif function_name == "exp":
            file.write(f'Curve fit weights: a = {m} and b = {n}.\n')
        elif function_name == "pow4":
            file.write(f'Curve fit weights: a = {m}, b = {n}, c = {o} and d = {m1}.\n')
        file.write(
            f'A model accuracy of {max_acc_gd_weighted} is predicted for {pred_sample_size} samples.\n')
        file.write(f'The mae for the curve fit is {output_gd_weighted}.\n\n')


def fit_and_predict(train_acc, sample_sizes, pred_sample_size, n_epochs):
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define mean squared error cost and exponential curve fit functions
    mse = nn.MSELoss()

    def exp_func(x, a, b):
        return a * (x ** b)

    # Define variables, learning rate and number of epochs for fitting with TF
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    learning_rate = 0.01
    training_epochs = 5000

    # Fit the exponential function to the data
    for epoch in range(training_epochs):
        y_pred = exp_func(torch.tensor(x, dtype=torch.float32), a, b)
        cost_function = mse(y_pred, torch.tensor(mean_acc, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            a -= a.grad * learning_rate
            b -= b.grad * learning_rate
            a.grad.zero_()
            b.grad.zero_()

    print(f"Curve fit weights: a = {a} and b = {b}.")
    # We can now estimate the accuracy for pred_sample_size
    max_acc = exp_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b).detach().numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, exp_func(torch.tensor(x_cont, dtype=torch.float32), a, b).detach().numpy(), "r-",
            label="Fitted exponential curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/subsample_size_{n_epochs}epochs.png')
    mae = nn.L1Loss()
    output = mae(exp_func(torch.tensor(x, dtype=torch.float32), a, b), torch.tensor(mean_acc, dtype=torch.float32))
    print(f"The mae for the curve fit is {output}.")
    with open(output_results_filename + str(n_epochs) + 'epochs.txt', 'a+') as f:
        f.write(f'fit_and_predict:\n')
        f.write(f'Curve fit weights: a = {a} and b = {b}.\n')
        f.write(f'A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.\n')
        f.write(f'The mae for the curve fit is {output}.\n\n')


def fit_and_predict_gd_unweighted(train_acc, sample_sizes, pred_sample_size, n_epochs):
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define mean squared error cost and inverse power law curve fit functions
    mse = nn.MSELoss()

    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    # Define variables, learning rate and number of epochs for fitting with TF
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.01
    training_epochs = 5000

    # Fit the exponential function to the data
    for epoch in range(training_epochs):
        y_pred = inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c)
        cost_function = mse(y_pred, torch.tensor(mean_acc, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            a -= a.grad * learning_rate
            b -= b.grad * learning_rate
            c -= c.grad * learning_rate
            a.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()

    print(f"Curve fit weights: a = {a}, b = {b} and c = {c}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = inverse_power_law_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b, c).detach().numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples with gd_unweighted.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), a, b, c).detach().numpy(), "r-",
            label="Fitted inverse power law curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/subsample_size_gd_unweighted_{n_epochs}epochs.png')
    mae = nn.L1Loss()
    output = mae(inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c),
                 torch.tensor(mean_acc, dtype=torch.float32))
    print(f"The mae for the curve fit is {output}.")
    with open(output_results_filename + str(n_epochs) + 'epochs.txt', 'a+') as f:
        f.write(f'fit_and_predict_gd_unweighted:\n')
        f.write(f'Curve fit weights: a = {a}, b = {b} and c = {c}.\n')
        f.write(f'A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.\n')
        f.write(f'The mae for the curve fit is {output}.\n\n')


def fit_and_predict_gd_weighted(train_acc, sample_sizes, pred_sample_size, n_epochs):
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define inverse power law curve fit and custom loss functions
    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    def custom_loss_function(y_pred, y_true, sigma):
        squared_difference = ((y_pred - y_true) ** 2) * sigma
        loss = torch.mean(squared_difference)
        return loss

    # Define variables, learning rate and number of epochs for fitting with TF
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.01
    training_epochs = 5000

    sigma = np.ones(len(x))
    start_index = 0
    stop_index = len(sigma)
    step = 1
    # applying for loop
    for i in range(start_index, stop_index, step):
        sigma[i] = (i + 1) / len(sigma)

    # Fit the exponential function to the data
    for epoch in range(training_epochs):
        y_pred = inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c)
        cost_function = custom_loss_function(y_pred,
                                             torch.tensor(mean_acc, dtype=torch.float32),
                                             torch.tensor(sigma, dtype=torch.float32))
        cost_function.backward()
        with torch.no_grad():
            a -= a.grad * learning_rate
            b -= b.grad * learning_rate
            c -= c.grad * learning_rate
            a.grad.zero_()
            b.grad.zero_()
            c.grad.zero_()

    print(f"Curve fit weights: a = {a}, b = {b} and c = {c}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = inverse_power_law_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b, c).detach().numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples with gd_weighted.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), a, b, c).detach().numpy(), "r-",
            label="Fitted inverse power law curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/subsample_size_gd_weighted_{n_epochs}epochs.png')
    mae = nn.L1Loss()
    output = mae(inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c),
                 torch.tensor(mean_acc, dtype=torch.float32))
    print(f"The mae for the curve fit is {output}.")
    with open(output_results_filename + str(n_epochs) + 'epochs.txt', 'a+') as f:
        f.write(f'fit_and_predict_gd_weighted:\n')
        f.write(f'Curve fit weights: a = {a}, b = {b} and c = {c}.\n')
        f.write(f'A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.\n')
        f.write(f'The mae for the curve fit is {output}.\n\n')


def fit_and_predict_nls_unweighted(train_acc, sample_sizes, pred_sample_size, n_epochs):
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define inverse power law curve fit function
    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    sample_sizes_array = np.array(sample_sizes)
    mean_acc_array = np.array(mean_acc)
    popt, _ = curve_fit(inverse_power_law_func, sample_sizes_array, mean_acc_array, maxfev=1000, bounds=(-7, 7))
    a, b, c = popt

    print(f"Curve fit weights: a = {a}, b = {b} and c = {c}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = inverse_power_law_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b, c).detach().numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples with nls_unweighted.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), a, b, c).detach().numpy(), "r-",
            label="Fitted inverse power law curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/subsample_size_nls_unweighted_{n_epochs}epochs.png')
    mae = nn.L1Loss()
    output = mae(inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c),
                 torch.tensor(mean_acc, dtype=torch.float32))
    print(f"The mae for the curve fit is {output}.")
    with open(output_results_filename + str(n_epochs) + 'epochs.txt', 'a+') as f:
        f.write(f'fit_and_predict_nls_unweighted:\n')
        f.write(f'Curve fit weights: a = {a}, b = {b} and c = {c}.\n')
        f.write(f'A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.\n')
        f.write(f'The mae for the curve fit is {output}.\n\n')


def fit_and_predict_nls_weighted(train_acc, sample_sizes, pred_sample_size, n_epochs):
    x = sample_sizes
    mean_acc = [np.mean(i) for i in train_acc]
    error = [np.std(i) for i in train_acc]

    # Define inverse power law curve fit function
    def inverse_power_law_func(x, a, b, c):
        return (1 - a) - b * (x ** c)

    sample_sizes_array = np.array(sample_sizes)
    mean_acc_array = np.array(mean_acc)
    sigma = np.ones(len(x))

    start_index = len(sigma)
    stop_index = 0
    step = -1
    count = 0

    # applying for loop
    for i in range(start_index, stop_index, step):
        sigma[count] = i / len(sigma)
        count = count + 1

    popt, _ = curve_fit(inverse_power_law_func, sample_sizes_array, mean_acc_array,
                        sigma=sigma, maxfev=1000, bounds=(-7, 7))
    a, b, c = popt

    print(f"Curve fit weights: a = {a}, b = {b} and c = {c}.")

    # We can now estimate the accuracy for pred_sample_size
    max_acc = inverse_power_law_func(torch.tensor(pred_sample_size, dtype=torch.float32), a, b, c).detach().numpy()

    # Print predicted x value and append to plot values
    print(f"A model accuracy of {max_acc} is predicted for {pred_sample_size} samples with nls_weighted.")
    x_cont = np.linspace(x[0], pred_sample_size, 100)

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(x, mean_acc, yerr=error, fmt="o", label="Mean acc & std dev.")
    ax.plot(x_cont, inverse_power_law_func(torch.tensor(x_cont, dtype=torch.float32), a, b, c).detach().numpy(), "r-",
            label="Fitted inverse power law curve.")
    ax.set_ylabel("Model clasification accuracy.", fontsize=12)
    ax.set_xlabel("Training sample size.", fontsize=12)
    ax.set_xticks(np.append(x, pred_sample_size))
    ax.set_yticks(np.append(mean_acc, max_acc))
    ax.set_xticklabels(list(np.append(x, pred_sample_size)), rotation=90, fontsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_title("Learning curve: model accuracy vs sample size.", fontsize=14)
    ax.legend(loc=(0.75, 0.75), fontsize=10)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/subsample_size_nls_weighted_{n_epochs}epochs.png')
    mae = nn.L1Loss()
    output = mae(inverse_power_law_func(torch.tensor(x, dtype=torch.float32), a, b, c),
                 torch.tensor(mean_acc, dtype=torch.float32))
    print(f"The mae for the curve fit is {output}.")
    with open(output_results_filename + str(n_epochs) + 'epochs.txt', 'a+') as f:
        f.write(f'fit_and_predict_nls_weighted:\n')
        f.write(f'Curve fit weights: a = {a}, b = {b} and c = {c}.\n')
        f.write(f'A model accuracy of {max_acc} is predicted for {pred_sample_size} samples.\n')
        f.write(f'The mae for the curve fit is {output}.\n\n')

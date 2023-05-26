# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import copy
import math
import pdb
import logging
import random
import csv

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from torchtext.legacy.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.legacy import data

from nas.nas_arch import *
from nas.model_profiler import *

logger = logging.getLogger(__name__)


bptt = 100

TEXT = data.Field(
    tokenize = 'spacy',
    tokenizer_language = 'en_core_web_sm',
    fix_length=bptt
)
LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


# build vocab
MAX_VOCAB_SIZE = 30520
TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")


BATCH_SIZE = 64

train_iter, valid_iter, test_iter = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc



def search(model_arch):

    input_dim = len(TEXT.vocab)
    model_type = "transformer"
    model = build_model(model_arch=model_arch,model_type=model_type, input_dim=input_dim, nlabels=1)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    N_EPOCHS = 200
    first_zen_score = 0.

    if model_type=="transformer":
        src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    best_valid_loss = float('inf')

    zen_writer = open('zen.scores','w')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc, ave_zen_score, zen_score = train(model_type, model, train_iter, optimizer, criterion, src_mask)
        valid_loss, valid_acc = evaluate(model_type, model, valid_iter, criterion, src_mask)

        if epoch==0:
            first_zen_score = zen_score
            # return 0.,first_zen_score
        zen_writer.write(str(zen_score))

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Zen Score: {zen_score:5.2f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    model.load_state_dict(torch.load('tut1-model.pt'))

    test_loss, test_acc = evaluate(model_type, model, test_iter, criterion, src_mask)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

    zen_writer.close()

    return test_acc,first_zen_score



def train(model_type, model, iterator, optimizer, criterion, src_mask):

    epoch_loss = 0
    epoch_acc = 0
    zen_scores = 0
    first_zen_score = 0
    model.train()
    for i,batch in enumerate(iterator):
        optimizer.zero_grad()
        # import pdb; pdb.set_trace()
        if model_type=="transformer":
            predictions,zen_score = model(batch.text, src_mask, None)
        else:
            predictions = model(batch.text).squeeze(1)
            zen_score = torch.autograd.Variable(Tensor([0.]))
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        zen_scores += zen_score.item()
        if i==0:
            first_zen_score = zen_score
            print(f'zen score is {first_zen_score}')
            # return epoch_loss / len(iterator), epoch_acc / len(iterator), zen_scores / len(iterator), first_zen_score

    return epoch_loss / len(iterator), epoch_acc / len(iterator), zen_scores / len(iterator), first_zen_score



def evaluate(model_type, model, iterator, criterion, src_mask):

    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for i,batch in enumerate(iterator):
            if model_type=="transformer":
                predictions,zen_score = model(batch.text, src_mask, None)
            else:
                predictions = model(batch.text).squeeze(1)
                zen_score = -1
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def main():

    emsizes = list(range(1800,2000,50)) #[50,100,150,200,50]
    nhids = list(range(1800,2024,20)) #[50,100,500,100,100]
    nlayerss = list(range(1,4)) #[1,1,1,1,1,1]
    nheads = list(range(1,5,1)) #[1,1,1,1,1,1]
    zen_scores_tups = []

    # with open('results/nas.log','w') as nas_log:
    #     for emsize, nhid, nlayers, nhead in zip(emsizes,nhids,nlayerss,nheads):
    #         model_arch = {
    #             'emsize': emsize,
    #             'nhid': nhid,
    #             'nlayers': nlayers,
    #             'nhead': nhead,
    #         }
    #         best_perf,first_zen_score = search(model_arch)
    #         nas_log.write(f'best_perf: {best_perf}, zen_score: {first_zen_score}, emsize: {emsize}, nhid: {nhid}, nlayers: {nlayers}, nhead: {nhead}\n')

    # search free ranges
    # with open('results/nas.log','w') as nas_log:
    #     for emsize in emsizes:
    #         for nhid in nhids:
    #             for nlayers in nlayerss:
    #                 for nhead in nheads:
    #                     if emsize % nhead != 0:
    #                         continue
    #                     model_arch = {
    #                         'emsize': emsize,
    #                         'nhid': nhid,
    #                         'nlayers': nlayers,
    #                         'nhead': nhead,
    #                     }
    #                     best_perf,first_zen_score = search(model_arch)
    #                     zen_scores_tups.append( (first_zen_score,f'emsize: {emsize}, nhid: {nhid}, nlayers: {nlayers}, nhead: {nhead}') )
    #                     nas_log.write(f'best_perf: {best_perf}, zen_score: {first_zen_score}, emsize: {emsize}, nhid: {nhid}, nlayers: {nlayers}, nhead: {nhead}\n')
    #
    # max_zen_tup = max(zen_scores_tups,key=lambda item:item[0])
    # print(max_zen_tup)

    emsize = 1800
    nhid = 50
    nlayers = 5
    nhead = 5
    model_arch = {
        'emsize': emsize,
        'nhid': nhid,
        'nlayers': nlayers,
        'nhead': nhead,
    }
    best_perf,first_zen_score = search(model_arch)
    zen_scores_tups.append( (first_zen_score,f'emsize: {emsize}, nhid: {nhid}, nlayers: {nlayers}, nhead: {nhead}') )

    max_zen_tup = max(zen_scores_tups,key=lambda item:item[0])
    print(max_zen_tup)

if __name__ == "__main__":
    main()

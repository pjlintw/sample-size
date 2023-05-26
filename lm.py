# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import pdb
import logging
import time
import csv

from typing import Optional, Any


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nas.nas_arch import *

logger = logging.getLogger(__name__)


import spacy
spacy_en = spacy.load("en_core_web_sm")
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

train_iter = WikiText2(split='train')
# tokenizer = get_tokenizer('spacy')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 64
eval_batch_size = 100
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 100
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


ntokens = len(vocab) # the size of vocabulary


def search(model_arch):

    model = build_model(model_arch=model_arch,model_type="transformer",input_dim=ntokens).to(device)
    lr = 2e-5 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs = 30 # The number of epochs
    best_model = None
    best_loss = float('inf')
    first_zen_score = 0.

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        zen_score = train(model,optimizer,scheduler,criterion,epoch)
        if epoch==1:
            first_zen_score = zen_score
            # return 0.,first_zen_score # early exit for search

        val_loss = evaluate(model, val_data, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

        if math.exp(val_loss)<best_loss:
            best_loss = math.exp(val_loss)

    return best_loss,first_zen_score


def train(model,optimizer,scheduler,criterion,epoch):

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    first_zen_score = 0.


    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()

        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output,zen_score = model(data, src_mask)
        if i==0:
            first_zen_score = zen_score
            print(f'zen score is {first_zen_score}')

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:1.8f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | zen-score {:5.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss), zen_score))
            total_loss = 0
            start_time = time.time()

        # if i==0:
        #     break

    return first_zen_score


def evaluate(eval_model, data_source, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = eval_model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output,zen_score = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def main():

    emsizes = list(range(1800,2000,50)) #[50,100,150,200,50]
    nhids = list(range(1800,2024,20)) #[50,100,500,100,100]
    nlayerss = list(range(1,2)) #[1,1,1,1,1,1]
    nheads = list(range(1,2,1)) #[1,1,1,1,1,1]

    # best_perf: 0.0, zen_score: 8.643622398376465, emsize: 1950, nhid: 1820, nlayers: 1, nhead: 1

    zen_scores_tups = []

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

    # search in specified ranges
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

    #emsize: 1950, nhid: 1820, nlayers: 1, nhead: 1
    emsize = 1950
    nhid = 1820
    nlayers = 1
    nhead = 1
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

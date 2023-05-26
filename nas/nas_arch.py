# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import pdb
import logging

from typing import Optional, Any

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.autograd.functional import jacobian



logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def build_transformer(model_arch,input_dim=-1,nlabels=-1, activation="relu"):
    emsize = model_arch['emsize'] # 50 # embedding dimension
    nhid = model_arch['nhid'] #100 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = model_arch['nlayers'] #2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = model_arch['nhead'] #2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(input_dim, emsize, nhead, nhid, nlayers, dropout, nlabels, activation).to(device)
    return model


def build_rnn(input_dim=-1):
    embedding_dim = 50
    hidden_dim = 50
    output_dim = 1
    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim).to(device)
    return model


def build_model(model_type=None,model_arch=None,input_dim=None,nlabels=-1, activation="relu"):
    if model_type=='transformer':
        model = build_transformer(model_arch=model_arch,input_dim=input_dim,nlabels=nlabels, activation=activation)
    elif model_type=='rnn':
        model = build_rnn(input_dim=input_dim)
    else:
        model = build_transformer(model_arch=model_arch,input_dim=input_dim,nlabels=nlabels, activation=activation)
    return model


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):

        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):

        #text = [sent len, batch size]
        embedded = self.embedding(text)
        #embedded = [sent len, batch size, emb dim]
        output, hidden = self.rnn(embedded)
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        src = torch.rand(10, 32, 512)
        out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.sigmas, self.final_sigma = [], -1

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        self.sigmas = []
        for mod in self.layers:
            output,sigmas = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            self.sigmas.extend(sigmas)

        """
            Compute zen-score
        """
        new_sigmas = []
        for sigma in self.sigmas:
            C_dim = sigma.size()[0]
            new_sigmas.append(torch.sqrt( torch.sum(torch.pow(sigma, 2)) / C_dim))

        x = torch.tensor(new_sigmas)
        # norm = x.norm(p=2, dim=0, keepdim=True)
        # m = nn.ReLU()
        # normalized_sigmas = m(x)

        # import pdb; pdb.set_trace()

        self.final_sigma = torch.sum(torch.log(x)) #torch.prod(x)  #torch.sum(torch.log(normalized_sigmas))

        # if self.norm is not None:
        #     output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
    Examples::
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(10, 32, 512)
        out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        src = torch.rand(32, 10, 512)
        out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm1 = ZenBatchNorm(d_model,True).to(device)
        # self.norm2 = ZenBatchNorm(d_model,True).to(device)

        self.norm1 = ZenLayerNorm(d_model).to(device)
        self.norm2 = ZenLayerNorm(d_model).to(device)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = self.activation  # F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src, sigma1 = self.norm1(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src, sigma2 = self.norm2(src)
        sigmas = [ sigma1 ]#,sigma2]
        return src, sigmas


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, nlabels=-1, activation="relu"):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.nlabels = nlabels
        self.out_dim = nlabels if nlabels>0 else ntoken
        print(f"out_dim: {self.out_dim}")
        # TODO add right decoder
        self.decoder = nn.Linear(ninp, self.out_dim)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.out_dim, nhead=1)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.zen_score = -1

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):

        def _init_weights_(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(0.)

        initrange = 0.1
        norm_0,norm_1 = 0.,1.
        self.encoder.weight.data.uniform_(norm_0, norm_1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(norm_0, norm_1)
        # self.transformer_encoder.apply(_init_weights_)
        # self.transformer_decoder.apply(_init_weights_)

    def forward(self, src, src_mask, tgt=None, is_zen=False):

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)

        if self.training and is_zen:
            # make leaf variables out of the inputs
            x_ = Variable(src.data,requires_grad=True)
            self.transformer_encoder(x_, src_mask).mean().backward()
            # print(torch.log(x_.grad.mean()))
            # print(self.transformer_encoder.final_sigma)            
            # self.zen_score = self.transformer_encoder.final_sigma + x_.grad.mean() # seems to work
            # self.zen_score_v1 = self.transformer_encoder.final_sigma + torch.log(x_.grad.mean())            
            
            self.zen_score = torch.log(x_.grad.mean()) # best so far
            # self.zen_score = torch.log(x_.grad.mean()) # best so far
            # self.zen_score = torch.log(self.transformer_encoder.final_sigma)

            # self.zen_score = torch.mean(torch.flatten(jacobian(self.decoder, memory)))                
            # print(self.zen_score)
            self.zen_score = torch.norm(torch.flatten(jacobian(self.decoder, memory))) 
        else:
            # make leaf variables out of the inputs
            # x_ = Variable(src.data,requires_grad=True)
            # self.transformer_encoder(x_, src_mask).mean().backward()
            # self.zen_score = torch.log(self.transformer_encoder.final_sigma)
            self.zen_score = torch.norm(torch.flatten(jacobian(self.decoder, memory))) 

        # TODO add correct decoder
        if tgt!=None:
            decoder_output = self.transformer_decoder(tgt, memory)
        else:            
            decoder_output = self.decoder(memory)

        # for sequence predictions
        if self.nlabels>0:
            # decoder_output = torch.sum(decoder_output,dim=0).squeeze(dim=0).squeeze(dim=1)
            decoder_output = torch.sum(decoder_output, dim=0).squeeze(dim=0)
            if decoder_output.dim() > 1:
                decoder_output = decoder_output.squeeze(dim=1)

        return decoder_output, self.zen_score


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ZenLayerNorm(nn.Module):

    def __init__(self, eps = 0.00001):
        super(ZenLayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        """
            Dimensions (L,N,C)
        """
        # import pdb; pdb.set_trace()
        mean = torch.mean(x, dim=2, keepdim=True)
        var = torch.square(x - mean).mean(dim=2, keepdim=True)
        sigma = var.view(-1)
        return (x - mean) / torch.sqrt(var + self.eps), sigma


class ZenBatchNorm(nn.BatchNorm1d):
    def forward(self, x):

        # import pdb; pdb.set_trace()

        x = x.permute(1,2,0)   # (L,N,C) -> (N,C,L) [20, 200, 35]
        self._check_input_dim(x)
        y = x
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                    self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
            y = y - mu.view(-1,1)
            y = y / (sigma2.view(-1,1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)

        return y.view(return_shape).permute(2,0,1), sigma2


from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
from args import *
#from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
#from scipy.stats import pearsonr, spearmanr
#from sklearn.metrics import matthews_corrcoef, f1_score
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert import *
#from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import unicodedata
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
import torch.nn.functional as F
import random

from collections import Counter


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, label=None, text_syntax=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.text_syntax = text_syntax
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, syntax_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.syntax_ids = syntax_ids

class InputFeatures_syntax(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, syntax_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.syntax_ids = syntax_ids
        
class DataProcessor(object):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""       
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_syntax_word.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid_syntax_word.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_syntax_word.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0","1","2","3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0].split(' ')
            text_syntax = line[1].split(' ') 
            label = line[-1] 
            assert len(text_a) == len(text_syntax)
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, text_syntax=text_syntax))
        return examples

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def judge_ignore(word):
    if len(_clean_text(word)) == 0:
        return True
    for char in word:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            return True
    return False

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

class Vocab(object):
    def __init__(self, bert_vocab_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_vocab_path, do_lower_case=False
        )

    def convert_tokens_to_ids(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor(token_ids, dtype=torch.long)
        mask = torch.ones(len(ids), dtype=torch.long)
        return ids, mask

    def subword_tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        token_start_idxs = torch.cumsum(torch.tensor([0] + subword_lengths[:-1]), dim=0)
        return subwords, token_start_idxs

    def subword_tokenize_to_ids(self, tokens):
        tokens = ["[PAD]" if judge_ignore(t) else t for t in tokens]
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(len(subword_ids), dtype=torch.uint8)
        token_starts[token_start_idxs] = 1
        return subword_ids, mask, token_starts

    def tokenize(self, tokens):
        subwords = list(map(self.tokenizer.tokenize, tokens))
        subword_lengths = [1] + list(map(len, subwords)) + [1]
        subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
        return subwords

class Field(object):

    def __init__(self, name, pad=0, unk=None, bos='[CLS]', sep='[SEP]',
                 lower=False, use_vocab=True, tokenizer=None, fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.sep = sep
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenizer = tokenizer
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, sep]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.sep is not None:
            params.append(f"sep={self.sep}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        return self.specials.index(self.sep)

    def transform(self, sequence):
        if self.tokenizer is not None:
            sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        if self.lower:
            sequence = [str.lower(token) for token in sequence]
        if self.fn is not None:
            sequence = [self.fn(token) for token in sequence]
        return sequence

    def numericalize(self, sequences):
        sequences = [self.transform(sequence) for sequence in sequences]
        # if self.use_vocab:
        #     sequences = [self.vocab.token2id(sequence)
        #                  for sequence in sequences]
        # if self.bos:
        #     sequences = [[self.bos_index] + sequence for sequence in sequences]
        # if self.sep:
        #     sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]
        return sequences

class BertField(Field):

    def numericalize(self, sequences):
        subwords, lens = [], []
        sequences = [([self.bos] if self.bos else []) + list(sequence) +
                     ([self.sep] if self.sep else [])
                     for sequence in sequences]

        for sequence in sequences:
            sequence = [self.transform(token) for token in sequence]
            sequence = [piece if piece else self.transform(self.pad)
                        for piece in sequence]
            subwords.append(sum(sequence, []))
            lens.append(torch.tensor([len(piece) for piece in sequence]))
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).ge(0) for pieces in subwords]
        return (subwords, lens, mask)

class BertEmbedding(nn.Module):

    def __init__(self, model, requires_grad=True):
        super(BertEmbedding, self).__init__()

        self.bert = BertModel.from_pretrained(model)
        #self.bert = self.bert.requires_grad_(requires_grad)
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"n_layers={self.n_layers}, n_out={self.n_out}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'
        return s

    def forward(self, subwords, bert_lens, bert_mask):
        batch_size, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_mask = bert_mask.gt(0)
        if not self.requires_grad:
            self.bert.eval()
        bert, _ = self.bert(subwords, attention_mask=bert_mask, output_all_encoded_layers=False)
        bert = bert[bert_mask].split(bert_lens[mask].tolist())
        bert = torch.stack([i.mean(0) for i in bert])
        bert_embed = bert.new_zeros(batch_size, seq_len, self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(-1), bert)
        return bert_embed

class SyntaxField(object):
    def __init__(self, pad=0, unk='<UNK>', bos='<CLS>', sep='<SEP>'):
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.sep = sep
        
        label_list=['<PAD>','<UNK>','<CLS>','<SEP>','<ROOT>','ADV','AMOD','APP','AUX','BNF','CJT','CJTN','CJTN0','CJTN1','CJTN2','CJTN3','CJTN4','CJTN5','CJTN6','CJTN7','CJTN8','CJTN9','CND','COMP','DIR','DMOD','EXT','FOC','IO','LGS','LOC','MNR','NMOD','OBJ','OTHER','PRD','PRP','PRT','RELC','ROOT','SBJ','TMP','TPC','UNK','VOC','cCJTN']
        self.syntax_dict={}
        for item in label_list:
            self.syntax_dict[item]=len(self.syntax_dict)
        self.len_syntax_dict = len(self.syntax_dict)

    def numericalize(self, sequences):
        seqids, lens = [], []
        sequences = [([self.bos] if self.bos else []) + list(sequence) +
                     ([self.sep] if self.sep else [])
                     for sequence in sequences]
        for seq in sequences:
            seq = [self.syntax_dict.get(label,self.syntax_dict.get(self.unk)) for label in seq]
            seqids.append(seq)
            lens.append(len(seq))
        return seqids

class DataLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def get_batches(self, subwords, lens, mask, labels, syntaxs=None, shuffle=False, hidden_embed=False):
        if syntaxs:
            batches_subwords, batches_lens, batches_mask, batches_labels, batches_syntaxs = [], [], [], [], []
        else:
            batches_subwords, batches_lens, batches_mask, batches_labels = [], [], [], []
        indexs = [i for i in range(len(subwords))]
        if shuffle:
            random.shuffle(indexs)
            subwords = [subwords[i] for i in indexs]
            lens = [lens[i] for i in indexs]
            mask = [mask[i] for i in indexs]
            labels = [labels[i] for i in indexs]
            if syntaxs:
                # from pdb import set_trace
                # set_trace()
                try:
                    syntaxs = [syntaxs[i] for i in indexs]
                except:
                    from pdb import set_trace
                    set_trace()
        start = 0
        start_indexs = []
        while start <= len(indexs)-1:
            start_indexs.append(start)
            start += self.batch_size

        for start in start_indexs:
            cur_indexs = indexs[start:start + self.batch_size]
            cur_subwords = [subwords[i] for i in cur_indexs]
            cur_lens = [lens[i] for i in cur_indexs]
            cur_labels = [labels[i] for i in cur_indexs]
            cur_mask = [mask[i] for i in cur_indexs]
            if syntaxs:
                cur_syntaxs = [syntaxs[i] for i in cur_indexs]
            maxlen_i, maxlen_j, maxlen_k = 0, 0, 0
            for i, j, k in zip(cur_subwords, cur_lens, cur_mask):
                maxlen_i, maxlen_j, maxlen_k = max(maxlen_i, len(i)), max(maxlen_j, len(j)), max(maxlen_k, len(k))
            batch_a, batch_b, batch_c = [], [], []
            for a, b, c in zip(cur_subwords, cur_lens, cur_mask):
                batch_a.append([i for i in a]+[0]*(maxlen_i-len(a)))
                batch_b.append([i for i in b]+[0]*(maxlen_j-len(b)))
                batch_c.append([i for i in c]+[0]*(maxlen_k-len(c)))
            if syntaxs:
                batch_syntax = []
                if not hidden_embed:
                    for d in cur_syntaxs:
                        batch_syntax.append([i for i in d]+[0]*(maxlen_j-len(d)))
                else:
                    hidden_size = cur_syntaxs[0].shape[1]
                    batch_syntax = torch.zeros(min(self.batch_size,len(cur_syntaxs)), maxlen_j, hidden_size)
                    for i, matrix in enumerate(cur_syntaxs):
                        one_len, _ = matrix.shape
                        # from pdb import set_trace
                        # set_trace()
                        batch_syntax[i, :one_len] = matrix[: , :]  
            batch_label = cur_labels
            batches_subwords.append(torch.LongTensor(batch_a))
            batches_lens.append(torch.LongTensor(batch_b))
            batches_mask.append(torch.LongTensor(batch_c))
            batches_labels.append(torch.LongTensor(batch_label))
            if syntaxs:
                if hidden_embed:
                    batches_syntaxs.append(batch_syntax)
                else:
                    batches_syntaxs.append(torch.LongTensor(batch_syntax))
        if syntaxs:
            return [item for item in zip(batches_subwords, batches_lens, batches_mask, batches_labels, batches_syntaxs)]
        else:
            return [item for item in zip(batches_subwords, batches_lens, batches_mask, batches_labels)]

class Bert_Classifier_Pooling(nn.Module):
    def __init__(self, data):
        super(Bert_Classifier_Pooling, self).__init__()
        self.bert_embedding = BertEmbedding(data.bert_path)
        self.args = data
        self.use_syntax = data.use_syntax
        if self.use_syntax:
            self.syntax_embed = nn.Embedding(data.len_syntax_dict, data.syntax_dim)
            self.lstm = nn.LSTM(data.bert_embedding_size+data.syntax_dim, data.hidden_dim, num_layers=data.lstm_layer, batch_first=True, bidirectional=data.bilstm)
        else:
            self.lstm = nn.LSTM(input_size=data.bert_embedding_size, hidden_size=data.hidden_dim, num_layers=data.lstm_layer, batch_first=True, bidirectional=data.bilstm)
        
        if data.bilstm:
            self.linear = nn.Linear(data.hidden_dim*2, data.num_labels)
        else:
            self.linear = nn.Linear(data.hidden_dim, data.num_labels)
        self.dropout = nn.Dropout(data.dropout)
     
    def forward(self, input_idxs, bert_lens, bert_mask, syntax_ids=None):
        bert_outs = self.bert_embedding(input_idxs, bert_lens, bert_mask)
        lens = torch.sum(bert_lens.gt(0), dim=1)
        # bert_outs = torch.split(bert_outs[token_start], lens.tolist())
        # bert_outs = pad_sequence(bert_outs, batch_first=True)
        lstm_input = bert_outs
        if self.use_syntax:
            syntax_vec = self.syntax_embed(syntax_ids)
            lstm_input = torch.cat((lstm_input, syntax_vec),-1)
        # max_len = lstm_input.size(1)
        # lstm_input = lstm_input[:, :max_len, :]
        # mask = torch.arange(max_len).cuda() < lens.unsqueeze(-1)
        # add lstm after bert
        sorted_lens, sorted_idx = torch.sort(lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        lstm_input = lstm_input[sorted_idx]
        lstm_input = pack(lstm_input, sorted_lens, batch_first=True, enforce_sorted=False)
        
        lstm_output, (h, _) = self.lstm(lstm_input) # lstm_output:[batch,sequence_length,embeding]
        
        output, _ = pad(lstm_output, batch_first=True)
        output = output.permute(0, 2, 1) # lstm_output:[batch,embeding,sequence_length]
        if self.args.maxpooling:
            output = F.max_pool1d(output, output.size()[2]) # lstm_output:[batch,embeding,1]
        elif self.args.avepooling:
            output = F.avg_pool1d(output, output.size()[2]) # lstm_output:[batch,embeding,1]
        output = output.squeeze(2) # lstm_output:[batch,embeding]
        output = output[reverse_idx]
        out = self.linear(torch.tanh(output))
        out = F.softmax(out,dim=1)
        return out
    
class Bert_Classifier_Pooling_hidden(nn.Module):
    def __init__(self, data, squeeze = False):
        super(Bert_Classifier_Pooling_hidden, self).__init__()
        self.bert_embedding = BertEmbedding(data.bert_path)
        self.args = data
        self.use_syntax = data.use_syntax
        if self.use_syntax:
            self.lstm = nn.LSTM(data.bert_embedding_size+data.syntax_dim, data.hidden_dim, num_layers=data.lstm_layer, batch_first=True, bidirectional=data.bilstm)
        else:
            self.lstm = nn.LSTM(input_size=data.bert_embedding_size, hidden_size=data.hidden_dim, num_layers=data.lstm_layer, batch_first=True, bidirectional=data.bilstm)
        
        # self.reduce = nn.Sequential(
        #     nn.Linear(nn.Linear(),
        #     nn.ReLU())
        # )

        if data.bilstm:
            self.linear = nn.Linear(data.hidden_dim*2, data.num_labels)
        else:
            self.linear = nn.Linear(data.hidden_dim, data.num_labels)
        self.dropout = nn.Dropout(data.dropout)
     
    def forward(self, input_idxs, bert_lens, bert_mask, syntax_embed=None):
        bert_outs = self.bert_embedding(input_idxs, bert_lens, bert_mask)
        lens = torch.sum(bert_lens.gt(0), dim=1)
        # bert_outs = torch.split(bert_outs[token_start], lens.tolist())
        # bert_outs = pad_sequence(bert_outs, batch_first=True)
        lstm_input = bert_outs
        if self.use_syntax:
            syntax_vec = syntax_embed
            lstm_input = torch.cat((lstm_input, syntax_vec), -1)
        # max_len = lstm_input.size(1)
        # lstm_input = lstm_input[:, :max_len, :]
        # mask = torch.arange(max_len).cuda() < lens.unsqueeze(-1)
        # add lstm after bert
        sorted_lens, sorted_idx = torch.sort(lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        lstm_input = lstm_input[sorted_idx]
        lstm_input = pack(lstm_input, sorted_lens, batch_first=True, enforce_sorted=False)
        
        lstm_output, (h, _) = self.lstm(lstm_input) # lstm_output:[batch,sequence_length,embeding]
        
        output, _ = pad(lstm_output, batch_first=True)
        output = output.permute(0, 2, 1) # lstm_output:[batch,embeding,sequence_length]
        if self.args.maxpooling:
            output = F.max_pool1d(output, output.size()[2]) # lstm_output:[batch,embeding,1]
        elif self.args.avepooling:
            output = F.avg_pool1d(output, output.size()[2]) # lstm_output:[batch,embeding,1]
        output = output.squeeze(2) # lstm_output:[batch,embeding]
        output = output[reverse_idx]
        out = self.linear(torch.tanh(output))
        out = F.softmax(out,dim=1)
        return out

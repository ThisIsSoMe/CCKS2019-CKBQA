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
from torch.nn import CosineSimilarity
from collections import Counter
from field import *

# def data_batchlize(batch_size, data_tuple, syntaxs=None, hidden_embed=False):
#     '''
#     give a tuple, return batches of data
#     '''
#     (subwords, lens, mask) = data_tuple
#     if syntaxs:
#         batches_subwords, batches_lens, batches_mask, batches_syntaxs = [], [], [], []
#     else:
#         batches_subwords, batches_lens, batches_mask = [], [], []
#     indexs = [i for i in range(len(subwords))]
#     start = 0
#     start_indexs = []
#     while start <= len(indexs)-1:
#         start_indexs.append(start)
#         start += batch_size
#     start = 0
#     for start in start_indexs:
#         cur_indexs = indexs[start:start + batch_size]
#         cur_subwords = [subwords[i] for i in cur_indexs]
#         cur_lens = [lens[i] for i in cur_indexs]
#         cur_mask = [mask[i] for i in cur_indexs]
#         if syntaxs:
#             cur_syntaxs = [syntaxs[i] for i in cur_indexs]
#         maxlen_i, maxlen_j, maxlen_k = 0, 0, 0
#         for i, j, k in zip(cur_subwords, cur_lens, cur_mask):
#             maxlen_i, maxlen_j, maxlen_k = max(maxlen_i, len(i)), max(maxlen_j, len(j)), max(maxlen_k, len(k))
#         batch_a, batch_b, batch_c = [], [], []
#         for a, b, c in zip(cur_subwords, cur_lens, cur_mask):
#             batch_a.append([i for i in a]+[0]*(maxlen_i-len(a)))
#             batch_b.append([i for i in b]+[0]*(maxlen_j-len(b)))
#             batch_c.append([i for i in c]+[0]*(maxlen_k-len(c)))
#         if syntaxs:
#             batch_syntax = []
#             if not hidden_embed:
#                 for d in cur_syntaxs:
#                     batch_syntax.append([i for i in d]+[0]*(maxlen_j-len(d)))
#             else:
#                 hidden_size = cur_syntaxs[0].shape[1]
#                 batch_syntax = torch.zeros(min(batch_size,len(cur_syntaxs)), maxlen_j, hidden_size)
#                 for i, matrix in enumerate(cur_syntaxs):
#                     one_len, _ = matrix.shape
#                     # from pdb import set_trace
#                     # set_trace()
#                     batch_syntax[i, :one_len] = matrix[: , :]  
#         batches_subwords.append(torch.LongTensor(batch_a))
#         batches_lens.append(torch.LongTensor(batch_b))
#         batches_mask.append(torch.LongTensor(batch_c))
#         if syntaxs:
#             if hidden_embed:
#                 batches_syntaxs.append(batch_syntax)
#             else:
#                 batches_syntaxs.append(torch.LongTensor(batch_syntax))
#     if syntaxs:
#         return [item for item in zip(batches_subwords, batches_lens, batches_mask, batches_syntaxs)]
#     else:
#         return [item for item in zip(batches_subwords, batches_lens, batches_mask)]

class BertEmbedding(nn.Module):

    def __init__(self, model, requires_grad=True):
        super(BertEmbedding, self).__init__()

        self.bert = BertModel.from_pretrained(model)
        #self.bert = self.bert.requires_grad_(requires_grad)
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size
        self.n_layers = 1

    def __repr__(self):
        s = self.__class__.__name__ + '('
        if hasattr(self, 'n_layers') and hasattr(self, 'n_out'):
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

class BertCharEmbedding(nn.Module):
    def __init__(self, path, requires_grad=True):
        super(BertCharEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.requires_grad = requires_grad
    
    def forward(self, subwords, bert_mask):
        bert, _ = self.bert(subwords, attention_mask=bert_mask, output_all_encoded_layers=False)
        return bert

class Bert_Comparing(nn.Module):
    def __init__(self, data):
        super(Bert_Comparing, self).__init__()

        self.question_bert_embedding = BertCharEmbedding(data.bert_path, data.requires_grad)
        self.path_bert_embedding = BertCharEmbedding(data.bert_path, data.requires_grad)
        self.args = data
        self.similarity = CosineSimilarity(dim=1)
    
    def question_encoder(self, input_idxs, bert_mask):
        bert_outs = self.question_bert_embedding(input_idxs, bert_mask)
        return bert_outs[:, 0]
    
    def path_encoder(self, input_idxs, bert_mask):
        bert_outs = self.path_bert_embedding(input_idxs, bert_mask)
        return bert_outs[:, 0]

    def forward(self, questions, pos, negs):
        '''
        questions: batch_size, max_seq_len

        pos_input_idxs: batch_size, max_seq_len
        pos_bert_lens: batch_size, max_seq_len
        pos_bert_mask: batch_size, max_seq_len

        neg_input_idxs: neg_size, batch_size, max_seq_len
        neg_bert_lens: neg_size, batch_size, max_seq_len
        neg_bert_mask: neg_size, batch_size, max_seq_len
        '''
        
        (q_input_idxs, q_bert_mask) = questions

        (pos_input_idxs, pos_bert_mask) = pos
        (neg_input_idxs, neg_bert_mask) = negs
        neg_size, batch_size, _ = neg_input_idxs.shape

        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size, hidden_dim)

        pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask)

        neg_input_idxs = neg_input_idxs.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)
        neg_bert_mask = neg_bert_mask.reshape(neg_size*batch_size, -1) # (neg_size*batch_size, max_seq_len)

        neg_encoding = self.path_encoder(neg_input_idxs, neg_bert_mask) # (neg_size*batch_size, hidden_dim)
        # p_encoding = p_encoding.reshape(neg_size, batch_size, -1) # (neg_size, batch_size, hidden_dim)
        
        q_encoding_expand = q_encoding.unsqueeze(0).expand(neg_size, batch_size, q_encoding.shape[-1]).reshape(neg_size*batch_size, -1) # (neg_size*batch_size, hidden_dim)

        pos_score = self.similarity(q_encoding, pos_encoding)
        pos_score = pos_score.unsqueeze(1) # (batch_size, 1)
        neg_score = self.similarity(q_encoding_expand, neg_encoding)
        neg_score = neg_score.reshape(neg_size,-1).transpose(0,1) # (batch_size, neg_size)

        return (pos_score, neg_score)
    
    @torch.no_grad()
    def cal_score(self, question, cands, pos=None):
        '''
        one question, several candidate paths
        question: (max_seq_len), (max_seq_len), (max_seq_len)
        cands: (batch_size, max_seq_len), (batch_size, max_seq_len), (batch_size, max_seq_len)
        '''
        question = (t.unsqueeze(0) for t in question)

        if self.args.no_cuda == False:
            question = (t.cuda() for t in question)

        (q_input_idxs, q_bert_mask) = question
        
        q_encoding = self.question_encoder(q_input_idxs, q_bert_mask) # (batch_size=1, hidden_dim)
        
        if pos:
            pos = (t.unsqueeze(0) for t in pos)
            if self.args.no_cuda == False:
                pos = (t.cuda() for t in pos)
            
            (pos_input_idxs, pos_bert_mask) = pos
            pos_encoding = self.path_encoder(pos_input_idxs, pos_bert_mask) # (batch_size=1, hidden_dim)
            pos_score = self.similarity(q_encoding, pos_encoding) # (batch_size=1) 

        all_scores = []

        for (batch_input_idxs, batch_bert_mask) in cands:
            if self.args.no_cuda ==False:
                batch_input_idxs, batch_bert_mask = batch_input_idxs.cuda(), batch_bert_mask.cuda()
            path_encoding = self.path_encoder(batch_input_idxs, batch_bert_mask) #(batch_size, hidden_dim)
            q_encoding_expand = q_encoding.expand_as(path_encoding)
            scores = self.similarity(q_encoding_expand, path_encoding) # (batch_size)
            for score in scores:
                all_scores.append(score)
        all_scores = torch.Tensor(all_scores)

        if pos:
            return pos_score.cpu(), all_scores.cpu()
        else:
            return all_scores.cpu()
        
class Bert_ShareComparing(Bert_Comparing):
    def __init__(self, data):
        super(Bert_ShareComparing, self).__init__(data)
        self.question_bert_embedding = BertCharEmbedding(data.bert_path, data.requires_grad)
        self.path_bert_embedding = self.question_bert_embedding
        self.args = data
        self.similarity = CosineSimilarity(dim=1)

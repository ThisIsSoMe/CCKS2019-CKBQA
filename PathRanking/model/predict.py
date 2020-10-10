from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import sys
import json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
logger = logging.getLogger(__name__)

# personal package
from field import *
from bert_function import *
from model import *
from args import get_args
from data import Data

alpha = 0.1 # 字符层面得分占比 not 0.5

def occupied(seqa,seqb_list):
    # seqa:question
    # seqb:path
    # return value:[-1,1]
    scores = []
    for seqb in seqb_list:
        s = jaccard(seqa,seqb)
        scores.append(s)
    return scores

def jaccard(seqa,seqb):

    """
    返回两个句子的 jaccard 相似度 并没有 计算 字出现的次数
    """
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    bb = seqa.union(seqb)
    #return (len(aa)-1)/len(bb)
    return len(aa)/len(bb)

def hint(seqa,seqb):
    seqa = set(list(seqa.upper()))
    seqb = set(list(seqb.upper()))
    aa = seqa.intersection(seqb)
    return len(aa)

def data_batchlize(batch_size, data_tuple):
    '''
    give a tuple, return batches of data
    '''
    (subwords, mask) = data_tuple

    batches_subwords, batches_mask = [], []

    indexs = [i for i in range(len(subwords))]
    start = 0
    start_indexs = []
    while start <= len(indexs)-1:
        start_indexs.append(start)
        start += batch_size
    
    start = 0
    for start in start_indexs:
        cur_indexs = indexs[start:start + batch_size]
        cur_subwords = [subwords[i] for i in cur_indexs]
        cur_mask = [mask[i] for i in cur_indexs]

        maxlen_i, maxlen_j = 0, 0
        for i, j in zip(cur_subwords, cur_mask):
            maxlen_i, maxlen_j = max(maxlen_i, len(i)), max(maxlen_j, len(j))
        batch_a, batch_b = [], []
        for a, b in zip(cur_subwords, cur_mask):
            batch_a.append([i for i in a]+[0]*(maxlen_i-len(a)))
            batch_b.append([i for i in b]+[0]*(maxlen_j-len(b)))
        
        batches_subwords.append(torch.LongTensor(batch_a))
        batches_mask.append(torch.LongTensor(batch_b))

    return [item for item in zip(batches_subwords, batches_mask)]

# 删去实体中的带括号的描述信息
def del_des(string):
    stack=[]
    # if '_（' not in string and '）' not in string and '_(' not in string and ')' not in string:
    if '_' not in string:
        return string
    mystring=string[1:-1]
    if mystring[-1]!='）' and mystring[-1]!=')':
        return string
    for i in range(len(mystring)-1,-1,-1):
        char=mystring[i]
        if char=='）':
            stack.append('）')
        elif char == ')':
            stack.append(')')
        elif char=='（': 
            if stack[-1]=='）':
                stack=stack[:-1]
                if not stack:
                    break
        elif char=='(':
            if stack[-1]==')':
                stack=stack[:-1]
                if not stack:
                    break
    if mystring[i-1]=='_':
        i-=1
    else:
        return string
    return '<'+mystring[:i]+'>'

def predict(args, model, field):
    
    model.eval()
    Dataset =  Data(args)
    
    fn_in = args.input_file
    # if 'cand_paths' in fn_in:
    #     fn_out = fn_in.replace('cand_paths','best_path')
    # else:
    #     fn_out = fn_in.replace('paths','predict_path')
    if not args.output_file:
        fn_out = fn_in.replace('cand_paths','best_path')
    else:    
        fn_out = args.output_file

    with open(fn_in, 'r')as f:
        raw_data = json.load(f)
    
    output_data = {}

    topk = args.topk

    for line in raw_data:
        if 'q_ws' in line.keys():
            q, q_ws, paths, paths_ws = line['q'], line['q_ws'], line['paths'], line['paths_ws']
        else:
            q, paths= line['q'], line['paths']
        
        one_question = Dataset.numericalize(field, [q]) # 内部元素都是二维的
        one_question = [t[0] for t in one_question] # 内部是一维的

        one_question = (t for t in one_question)
        
        paths_input = [''.join([del_des(item) for item in path]) for path in paths]
        one_cands = Dataset.numericalize(field, paths_input)
        batches_cands = data_batchlize(args.test_batch_size, one_cands)
        
        # 字符层面得分
        char_scores = occupied(q,[''.join([del_des(i) for i in p]) for p in paths])
        char_scores = torch.Tensor(char_scores)
        # 模型层面得分
        model_scores = model.cal_score(one_question, batches_cands)
        all_scores = alpha*char_scores + (1-alpha)*model_scores

        if len(all_scores)>0 and topk == 1:
            index = torch.argmax(all_scores)
            output_data[q] = paths[index]
        elif len(all_scores)>0 and topk > 1:
            sorted_scores, index = torch.sort(all_scores, descending=True)
            output_data[q] = [paths[i] for i in index[:topk]]
        else:
            print(q, 'no path')
    
    with open(fn_out, 'w')as f:
        json.dump(output_data, f, ensure_ascii=False)

if __name__ == "__main__":

    args = get_args(mode='predict')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # tokenize
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)    
    bert_field = BertField('BERT', tokenizer=tokenizer)
    print("loaded tokenizer")

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('使用%s号GPU' % args.gpu)

    # model
    # model = Bert_Comparing(args)
    # if args.no_cuda == False:
    #     model.cuda()

    # Load a trained model that you have fine-tuned
    model_dict = {'bert_comparing':Bert_Comparing,'bert_sharecomparing':Bert_ShareComparing}
    model_name = model_dict[args.model]

    model_state_dict = torch.load(args.model_path)
    model = model_name(args)
    model.load_state_dict(model_state_dict)
    model.eval()

    if args.no_cuda == False:
        model.cuda()
    print('loaded model!')

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)    
    bert_field = BertCharField('BERT', tokenizer=tokenizer)
    print("loaded tokenizer!")

    fn = args.input_file
    with open(fn, 'r')as f:
        test_raw_data = json.load(f)
    predict(args, model, bert_field)

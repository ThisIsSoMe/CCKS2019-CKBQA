 # -*- encoding:utf-8 -*-

import json
import torch

class Data():
    def __init__(self, args):
        self.args = args

    def load(self, mode):
        if mode == 'train':
            with open(self.args.train_file, 'r') as f:
                my_data = json.load(f)
            out = (my_data['questions'], my_data['golds'],my_data['negs'])
        elif mode == 'valid':
            with open(self.args.valid_file, 'r') as f:
                my_data = json.load(f)
            out = (my_data['questions'], my_data['golds'],my_data['negs'])
        elif mode == 'test':
            with open(self.args.valid_file, 'r') as f:
                my_data = json.load(f)
            out = (my_data['questions'],my_data['cands'])
        return out
    
    def numericalize(self, field, seqs):
        out = field.numericalize(seqs)
        return out
    




import numpy as np
import sys
import pickle
import os
import random

START = "</s>"
PAD = "</pad>"
UNKNOWN = "</unk>"
MASK = "</mask>"

class Alphabet:
    def __init__(self, name, unknown_label=True,mask_label=False,threshold=0):
        self.name = name
        self.instance2index = {}
        self.instances = []
        # self.keep_growing = keep_growing
        self.index = 1
        self.threshold = threshold
        self.words = dict()
        if unknown_label:
            self.instances.append(UNKNOWN)
            self.instance2index[UNKNOWN] = self.index
            self.index += 1
        if mask_label:
            self.instances.append(MASK)
            self.instance2index[MASK] = self.index
            self.index += 1

    def add(self, instance):
        if instance not in self.instance2index:
            if instance not in self.words:
                self.words[instance] = 1
            else:
                self.words[instance] = self.words[instance]+1
            if self.words[instance] > self.threshold:
                self.instances.append(instance)
                self.instance2index[instance] = self.index
                self.index += 1

    def size(self):
        return len(self.instances) + 1

    def get_instance(self, index):
        try:
            return self.instances[index - 1]
        except:
            #print("Alphabet Warning ",index)
            return self.instances[0]

class Data:
    def __init__(self):
        self.config = {}
        self.train_text = []
        self.dev_text = []
        self.test_text = []
        self.oov_text = []
        self.train_idx = []
        self.train_mask_idx = []
        self.dev_idx = []
        self.test_idx = []
        self.oov_idx = []
        self.bichar_text = []
        self.bichar_idx = []

        self.word_alphabet = Alphabet("word",mask_label=True)
        self.char_alphabet = Alphabet("char")
        self.bichar_alphabet = Alphabet("bichar")
        self.label_alphabet = Alphabet("label", unknown_label=False)
        self.feature_alphabets = []
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_alphabet_size = []
        self.feature_names = []

        self.status = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None
        self.oov_file = None
        self.pretrain_file = None
        self.word_embed_path = None
        self.word_embed_save = None
        self.char_embed_path = None
        self.char_embed_save = None
        self.elmo_embed_path = None
        self.elmo_embed_save = None
        self.feature_embed_path = None
        self.feature_embed_save = None
        self.model_path = None
        self.model_save_dir = None
        self.result_save_dir = None
        self.dataset = None
        self.word_normalize = False
        self.word_feature_extractor = "LSTM"
        self.use_char = False
        self.char_feature_extractor = "LSTM"
        self.use_elmo = False
        self.use_crf = False
        self.use_cuda = False
        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_elmo_embedding = None
        self.pretrain_feature_embedding = None
        self.tag_scheme = None
        self.entity_mask = False
        self.mask_percent = 0

        # hyperparameters
        self.pretrain = False
        self.word_embed_dim = None
        self.char_embed_dim = None
        self.elmo_embed_dim = None
        self.feature_embed_dim = None
        self.word_seq_feature = None
        self.char_seq_feature = None
        self.optimizer = None
        self.hidden_dim = None
        self.char_hidden_dim = None
        self.bilstm = None
        self.lstm_layer = None
        self.batch_size = None
        self.dropout = None
        self.lr = None
        self.lr_decay = None
        self.momentum = None
        self.weight_decay = None
        self.iter = None
        self.fine_tune = False
        self.elmo_fine_tune = False
        self.attention = False
        self.lstm_attention = False
        self.attention_dim = None
        self.average_loss = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.number_normalized = False
        self.threshold = None
        self.max_sent_len = 0
        self.prob = None
        self.stopwords = None
        self.feature = None
        self.feature_num = 0
        self.out_dict = None
        self.out_dict_path = None
        self.hyperlstm = False
        self.hyper_hidden_dim = 0
        self.hyper_emb_dim = 0
        self.bert_path = None
        self.bert_vocab_path = None
        self.bert_embedding_size = 0
        self.use_bert = False
        self.train_bert_file = None
        self.dev_bert_file = None
        self.test_bert_file = None
        self.bert_dim = 0

    def get_instance(self, name):
        if name == 'train':
            self.read_data(self.train_file, name, self.dataset)
        elif name == 'dev':
            self.read_data(self.dev_file, name, self.dataset)
        elif name == 'test':
            self.read_data(self.test_file, name, self.dataset)
        elif name == 'oov':
            self.read_data(self.oov_file, name, self.dataset)
        else:
            print('Get Instance Error: pls set correct data instance.')

    def read_config(self, file):
        with open(file, 'r', encoding='utf-8') as fin:
            for line in fin:
                if len(line) > 0 and line[0] == '#':
                    continue
                if '=' in line:
                    parameter = line.strip().split('=',1)
                    if parameter[0] == "feature":
                        if parameter[0] not in self.config:
                            feat_dict = {}
                            self.config[parameter[0]] = feat_dict
                        feat_dict = self.config[parameter[0]]
                        new_pair = parameter[1].split()
                        one_dict = {}
                        one_dict['embed_dim'] = 0
                        one_dict['embed_path'] = None
                        one_dict['embed_save'] = None
                        one_dict['embed_norm'] = False
                        if len(new_pair) > 1:
                            for idx in range(1,len(new_pair)):
                                info = new_pair[idx].split('=')
                                if info[0] == 'embed_dim':
                                    one_dict['embed_dim'] = info[1]
                                elif info[0] == 'embed_path':
                                    one_dict['embed_path'] = info[1]
                                elif info[0] == 'embed_save':
                                    one_dict['embed_save'] = info[1]
                                elif info[0] == 'embed_norm':
                                    one_dict['embed_norm'] = str2bool(info[1])
                                else:
                                    raise RuntimeError('unrecognize feature config')
                        self.feature_names.append(new_pair[0])
                        feat_dict[new_pair[0]] = one_dict
                        self.feature = feat_dict
                    else:
                        if parameter[0] not in self.config:
                            self.config[parameter[0]] = parameter[1]
                        else:
                            print('Warning : duplicate parameter found.')
        if self.feature:
            self.feature_num = len(self.feature)
            self.pretrain_feature_embedding = [None]*self.feature_num
            self.feature_embed_dim = [0]*self.feature_num

        item = 'train_file'
        if item in self.config:
            self.train_file = self.config[item]
        item = 'dev_file'
        if item in self.config:
            self.dev_file = self.config[item]
        item = 'test_file'
        if item in self.config:
            self.test_file = self.config[item]
        item = 'oov_file'
        if item in self.config:
            self.oov_file = self.config[item]
        item = 'model_path'
        if item in self.config:
            self.model_path = self.config[item]
        item = 'model_save_dir'
        if item in self.config:
            self.model_save_dir = self.config[item]
        item = 'result_save_dir'
        if item in self.config:
            self.result_save_dir = self.config[item]
        item = 'dataset'
        if item in self.config:
            self.dataset = self.config[item]
        item = 'status'
        if item in self.config:
            self.status = self.config[item]
        item = 'word_feature_extractor'
        if item in self.config:
            self.word_feature_extractor = self.config[item]
        item = 'use_char'
        if item in self.config:
            self.use_char = str2bool(self.config[item])
        item = 'char_feature_extractor'
        if item in self.config:
            self.char_feature_extractor = self.config[item]
        item = 'use_crf'
        if item in self.config:
            self.use_crf = str2bool(self.config[item])
        item = 'use_elmo'
        if item in self.config:
            self.use_elmo = str2bool(self.config[item])
        item = 'word_embed_path'
        if item in self.config:
            self.word_embed_path = self.config[item]
        item = 'char_embed_path'
        if item in self.config:
            self.char_embed_path = self.config[item]
        item = 'elmo_embed_path'
        if item in self.config:
            self.elmo_embed_path = self.config[item]
        item = 'word_embed_dim'
        if item in self.config:
            self.word_embed_dim = int(self.config[item])
        item = 'char_embed_dim'
        if item in self.config:
            self.char_embed_dim = int(self.config[item])
        item = 'hidden_dim'
        if item in self.config:
            self.hidden_dim = int(self.config[item])
        item = 'char_hidden_dim'
        if item in self.config:
            self.char_hidden_dim = int(self.config[item])
        item = 'use_cuda'
        if item in self.config:
            self.use_cuda = str2bool(self.config[item])
        item = 'lr'
        if item in self.config:
            self.lr = float(self.config[item])
        item = 'lr_decay'
        if item in self.config:
            self.lr_decay = float(self.config[item])
        item = 'momentum'
        if item in self.config:
            self.momentum = float(self.config[item])
        item = 'weight_decay'
        if item in self.config:
            self.weight_decay = float(self.config[item])
        item = 'dropout'
        if item in self.config:
            self.dropout = float(self.config[item])
        item = 'iter'
        if item in self.config:
            self.iter = int(self.config[item])
        item = 'optimizer'
        if item in self.config:
            self.optimizer = self.config[item]
        item = 'batch_size'
        if item in self.config:
            self.batch_size = int(self.config[item])
        item = 'lstm_layer'
        if item in self.config:
            self.lstm_layer = int(self.config[item])
        item = 'pretrain'
        if item in self.config:
            self.pretrain = str2bool(self.config[item])
        item = 'fine_tune'
        if item in self.config:
            self.fine_tune = str2bool(self.config[item])
        item = 'elmo_fine_tune'
        if item in self.config:
            self.elmo_fine_tune = str2bool(self.config[item])
        item = 'attention'
        if item in self.config:
            self.attention = str2bool(self.config[item])
        item = 'lstm_attention'
        if item in self.config:
            self.lstm_attention = str2bool(self.config[item])
        item = 'attention_dim'
        if item in self.config:
            self.attention_dim = int(self.config[item])
        item = 'bilstm'
        if item in self.config:
            self.bilstm = str2bool(self.config[item])
        item = 'average_loss'
        if item in self.config:
            self.average_loss = str2bool(self.config[item])
        item = 'norm_word_emb'
        if item in self.config:
            self.norm_word_emb = str2bool(self.config[item])
        item = 'norm_char_emb'
        if item in self.config:
            self.norm_char_emb = str2bool(self.config[item])
        item = 'tag_scheme'
        if item in self.config:
            self.tag_scheme = self.config[item]
        item = 'word_embed_save'
        if item in self.config:
            self.word_embed_save = self.config[item]
        item = 'char_embed_save'
        if item in self.config:
            self.char_embed_save = self.config[item]
        item = 'elmo_embed_save'
        if item in self.config:
            self.elmo_embed_save = self.config[item]
        item = 'number_normalized'
        if item in self.config:
            self.number_normalized = str2bool(self.config[item])
        item = 'threshold'
        if item in self.config:
            self.threshold = int(self.config[item])
            self.word_alphabet.threshold = self.threshold
        item = 'max_sent_len'
        if item in self.config:
            self.max_sent_len = int(self.config[item])
        item = 'entity_mask'
        if item in self.config:
            self.entity_mask = str2bool(self.config[item])
        item = 'mask_percent'
        if item in self.config:
            self.mask_percent = float(self.config[item])
        item = 'stopwords'
        if item in self.config:
            self.stopwords = str2bool(self.config[item])
        item = 'out_dict_path'
        if item in self.config:
            self.out_dict_path = self.config[item]
        item = 'hyperlstm'
        if item in self.config:
            self.hyperlstm = str2bool(self.config[item])
        item = 'hyper_hidden_dim'
        if item in self.config:
            self.hyper_hidden_dim = int(self.config[item])
        item = 'hyper_emb_dim'
        if item in self.config:
            self.hyper_emb_dim = int(self.config[item])
        item = 'bert_path'
        if item in self.config:
            self.bert_path = self.config[item]
        item = 'bert_vocab_path'
        if item in self.config:
            self.bert_vocab_path = self.config[item]
        item = 'bert_embedding_size'
        if item in self.config:
            self.bert_embedding_size = int(self.config[item])
        item = 'use_bert'
        if item in self.config:
            self.use_bert = str2bool(self.config[item])
        item = 'train_bert_file'
        if item in self.config:
            self.train_bert_file = self.config[item]
        item = 'dev_bert_file'
        if item in self.config:
            self.dev_bert_file = self.config[item]
        item = 'test_bert_file'
        if item in self.config:
            self.test_bert_file = self.config[item]
        item = 'bert_dim'
        if item in self.config:
            self.bert_dim = int(self.config[item])

    def show_config(self):
        for k, v in self.config.items():
            print(k, v)

    def do_stopwords(self,sentence):
        stopwords = [line.strip() for line in open('', 'r', encoding='utf-8').readlines()]
        return sentence

    def read_data(self, path, data, datasetname):
        if path is None:
            return None
        samples,count,bert_vec = [],0,[]
        if self.use_bert:
            bert_file = ''
            if data == 'train':
                bert_file = self.train_bert_file
            elif data == 'dev':
                bert_file = self.dev_bert_file
            elif data == 'test':
                bert_file = self.test_bert_file
            if bert_file:
                bert_vec = self.read_bert_file(bert_file)
            print(len(bert_vec))
        with open(path, 'r', encoding='utf-8') as fin:
            if datasetname == 'conll2003':
                # fin.readline()
                word_instances = []
                feat_instances = []
                char_instances = []
                label_instances = []
                for line in fin:
                    if line.startswith("-DOCSTART-"):
                        fin.readline()
                        continue
                    if line != '\n':
                        info = line.rstrip().split()
                        word = info[0]
                        ner_tag = info[-1]
                        if self.number_normalized:
                            word = normalize_word(word)
                        word_instances.append(word)
                        char_instances.append(list(word))
                        label_instances.append(ner_tag)
                    else:
                        if len(word_instances) > self.max_sent_len:
                            word_instances,char_instances,label_instances = [],[],[]
                            continue
                        #assert len(word_instances) == bert_vec[count].size(0)
                        if self.use_bert:
                            if bert_vec:
                                samples.append([word_instances,feat_instances, char_instances, label_instances,bert_vec[count]])
                            else: 
                                samples.append([word_instances,feat_instances, char_instances, label_instances])
                        else:
                            samples.append([word_instances,feat_instances, char_instances, label_instances,[]])
                        #samples.append([word_instances,feat_instances,char_instances,label_instances])
                        word_instances = []
                        char_instances = []
                        label_instances = []
                        count += 1
                # samples.append([word_instances, feat_instances, char_instances, label_instances,bert_vec[count]])
            elif datasetname == 'ali':
                word_instances = []
                feat_instances = []
                char_instances = []
                label_instances = []
                for line in fin:
                    if line != '\n':
                        info = line.rstrip().split()
                        word = info[0]
                        ner_tag = info[-1]
                        if self.number_normalized:
                            word = normalize_word(word)
                        word_instances.append(word)
                        char_instances.append(list(word))
                        label_instances.append(ner_tag)
                        if self.feature:
                            assert(len(info)-2 == len(self.feature))
                            feat_list = []
                            for idx in range(len(self.feature)):
                                feat_list.append(info[idx+1])
                            feat_instances.append(feat_list)
                    else:
                        if 0< len(word_instances) <= self.max_sent_len:
                            samples.append([word_instances,feat_instances, char_instances, label_instances])
                        else:
                            left,right = 0,self.max_sent_len
                            while right <= len(word_instances):
                                samples.append([word_instances[left:right], feat_instances[left:right],char_instances[left:right], label_instances[left:right]])
                                left += self.max_sent_len
                                right += self.max_sent_len
                            if left < len(word_instances) and right > len(word_instances):
                                samples.append([word_instances[left:], feat_instances[left:],char_instances[left:],
                                                label_instances[left:]])
                        word_instances = []
                        char_instances = []
                        feat_instances = []
                        label_instances = []
                if len(word_instances) > 0:
                    samples.append([word_instances, feat_instances, char_instances, label_instances])
        if data == "train":
            self.train_text = samples
        elif data == "dev":
            self.dev_text = samples
        elif data == "test":
            self.test_text = samples
        elif data == "oov":
            self.oov_text = samples
        else:
            print("Data Error:pls set train/dev/test data parameter.")

    def read_bert_file(self, path):
        bert_vec = pickle.load(open(path,'rb'))
        return bert_vec

    def build_alphabet(self):
        feature_nums = len(self.feature_names)
        for idx in range(feature_nums):
            self.feature_alphabets.append(Alphabet(self.feature_names[idx]))
        for sample in self.train_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for feats in sample[1]:
                for idx,f in enumerate(feats):
                    self.feature_alphabets[idx].add(f)
            for char in sample[2]:
                for c in char:
                    self.char_alphabet.add(c)
            for label in sample[3]:
                self.label_alphabet.add(label)
        for sample in self.dev_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for feats in sample[1]:
                for idx,f in enumerate(feats):
                    self.feature_alphabets[idx].add(f)
            for char in sample[2]:
                for c in char:
                    self.char_alphabet.add(c)
            for label in sample[3]:
                self.label_alphabet.add(label)
        for sample in self.test_text:
            for word in sample[0]:
                self.word_alphabet.add(word)
            for feats in sample[1]:
                for idx,f in enumerate(feats):
                    self.feature_alphabets[idx].add(f)
            for char in sample[2]:
                for c in char:
                    self.char_alphabet.add(c)
            for label in sample[3]:
                self.label_alphabet.add(label)
               
        if self.out_dict_path:
            with open(self.out_dict_path,'r',encoding='utf-8') as fin:
                self.out_dict = {}
                for line in fin:
                    entity = line.strip().split()
                    if entity[0] not in self.out_dict:
                        self.out_dict[entity[0]] = entity[1]
        '''for sample in self.dev_text:
             for word in sample[0]:
                 self.word_alphabet.add(word)
             for char in sample[1][0]:
                 self.char_alphabet.add(char)
             for label in sample[2]:
                 self.label_alphabet.add(label)
        for sample in self.test_text:
             for word in sample[0]:
                 self.word_alphabet.add(word)
             for char in sample[1][0]:
                 self.char_alphabet.add(char)
             for label in sample[2]:
                 self.label_alphabet.add(label)'''
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        print("build alphabet finish.")


    def get_instance_index(self, data):
        instance_idx = []
        if data == "train":
            samples = self.train_text
        elif data == "dev":
            samples = self.dev_text
        elif data == "test":
            samples = self.test_text
        elif data == "oov":
            samples = self.oov_text
        for sample in samples:
            word_idx, feat_idx, char_idx, label_idx,text_vec = [], [], [],[],[]
            for word in sample[0]:
                if word in self.word_alphabet.instance2index:
                    word_idx.append(self.word_alphabet.instance2index[word])
                else:
                    word_idx.append(self.word_alphabet.instance2index[UNKNOWN])
            for feat in sample[1]:
                feats = []
                for idx,f in enumerate(feat):
                    if f in self.feature_alphabets[idx].instance2index:
                        feats.append(self.feature_alphabets[idx].instance2index[f])
                    else:
                        feats.append(self.feature_alphabets[idx].instance2index[UNKNOWN])
                feat_idx.append(feats)
            for char in sample[2]:
                chars = []
                for c in char:
                    if c in self.char_alphabet.instance2index:
                        chars.append(self.char_alphabet.instance2index[c])
                    else:
                        chars.append(self.char_alphabet.instance2index[UNKNOWN])
                char_idx.append(chars)
            for label in sample[3]:
                if label in self.label_alphabet.instance2index:
                    # if label == 'O':
                    #     if random.randint(1,10) <= 2:
                    #         label_idx.append(self.label_alphabet.instance2index[UNKNOWN])
                    #     else:
                    #         label_idx.append(self.label_alphabet.instance2index[label])
                    # else:
                    label_idx.append(self.label_alphabet.instance2index[label])
                else:
                    label_idx.append(self.label_alphabet.instance2index[UNKNOWN])
            if len(sample) == 5 :
                instance_idx.append([word_idx, feat_idx, char_idx, label_idx,sample[4]])
            else:
                instance_idx.append([word_idx, feat_idx, char_idx, label_idx])
        if data == "train":
            self.train_idx = instance_idx
            if self.entity_mask:
                self.train_mask_idx = list(self.train_idx)
                self.prob = [False]*len(self.train_idx)
        elif data == "dev":
            self.dev_idx = instance_idx
        elif data == "test":
            self.test_idx = instance_idx
        elif data == "oov":
            self.oov_idx = instance_idx
        else:
            print("Data Error:pls set train/dev/test data parameter.")

    def mask_entity(self,iter,num):
        mask_list = []
        if self.entity_mask:
            self.train_idx = list(self.train_mask_idx)
            length = len(self.train_idx)
            for idx in range(length):
                assert(len(self.train_idx[idx][-1]) == len(self.train_idx[idx][0]))
                if self.prob[idx] == False:
                    if random.random() < self.mask_percent or num+1 == iter:
                        label_items = self.train_idx[idx][3]
                        char_items = self.train_idx[idx][2]
                        feat_items = self.train_idx[idx][1]
                        word_items = self.train_idx[idx][0]
                        word = self.train_text[idx][0]
                        entities = []
                        entity,en_index = '',[]
                        find_entity = False
                        for idy,label in enumerate(label_items):
                            if label != self.label_alphabet.instance2index['O']:
                                word_items[idy] = self.word_alphabet.instance2index[MASK]
                        self.train_idx[idx][0] = word_items
                        mask_list.append([word_items,feat_items,char_items,label_items])
                        self.prob[idx] = True
        self.train_idx.extend(mask_list)

    def extract_non_verb(self,entities):
        mask_index = []
        for item in entities:
            result = HanLP.segment(item[0])
            # print(result)
            label_index = item[1]
            pos = 0
            info = str(result).strip('[').strip(']').split(',')
            for i in info:
                comb = i.split('/')
                if not comb[1].startswith('v'):
                    mask_index.extend(label_index[pos:pos+len(comb[0])])
                pos+=len(comb[0])
        return mask_index

    def build_pretrain_emb(self):
        if self.word_embed_path:
            self.pretrain_word_embedding, self.word_embed_dim = build_pretrain_embedding(self.word_embed_save,
                                                                                         self.word_embed_path,
                                                                                         self.word_alphabet,
                                                                                         self.word_embed_dim,
                                                                                         self.norm_word_emb)
        if self.char_embed_path:
            self.pretrain_char_embedding, self.char_embed_dim = build_pretrain_embedding(self.char_embed_save,
                                                                                         self.char_embed_path,
                                                                                         self.char_alphabet,
                                                                                         self.char_embed_dim,
                                                                                         self.norm_char_emb)
        if self.elmo_embed_path:
            self.pretrain_elmo_embedding, self.elmo_embed_dim = build_pretrain_embedding(self.elmo_embed_save,
                                                                                         self.elmo_embed_path,
                                                                                         self.word_alphabet,
                                                                                         self.word_embed_dim,
                                                                                         self.norm_word_emb)
        if self.feature:
            order = 0
            for k,v in self.feature.items():
                self.pretrain_feature_embedding[order], self.feature_embed_dim[order] = None,None
                if self.feature[k]["embed_dim"]:
                    self.feature_embed_dim[order] = int(self.feature[k]["embed_dim"])
                if self.feature[k]["embed_path"]:
                    self.pretrain_feature_embedding[order],self.feature_embed_dim[order] = build_pretrain_embedding(self.feature[k]["embed_save"],
                                                                                                            self.feature[k]["embed_path"],
                                                                                                            self.feature_alphabets[k],
                                                                                                            self.feature[k]["embed_dim"],
                                                                                                            self.feature[k]["embed_norm"])
                order += 1

    def word_alphabet_conll(self,out_path):
        with open(out_path,'w',encoding='utf-8') as fout:
            for (k,v) in self.word_alphabet.instance2index.items():
                fout.write(str(v)+'\t'+str(k)+'\t'+str(k)+'\t_\t_\t_\t_\t_\t_\t_\n')

    def show_data_summary(self):
        print("++"*50)
        print("DATA SUMMARY START:")
        print(" I/O:")
        print("     Tag          scheme: %s"%(self.tag_scheme))
        print("     MAX SENTENCE LENGTH: %s"%(self.max_sent_len))
        print("     Number   normalized: %s"%(self.number_normalized))
        print("     Word  alphabet size: %s"%(self.word_alphabet_size))
        print("     Char  alphabet size: %s"%(self.char_alphabet_size))
        print("     Label alphabet size: %s"%(self.label_alphabet_size))
        print("     Word embedding  path: %s"%(self.word_embed_path))
        print("     Word embedding  save: %s"%(self.word_embed_save))
        print("     Word embedding size: %s"%(self.word_embed_dim))
        print("     Char embedding  path: %s" % (self.char_embed_path))
        print("     Char embedding  save: %s" % (self.char_embed_save))
        print("     Char embedding size: %s" % (self.char_embed_dim))
        print("     Elmo embedding  path: %s" % (self.elmo_embed_path))
        print("     Elmo embedding  save: %s" % (self.elmo_embed_save))
        print("     Elmo embedding size: %s" % (self.elmo_embed_dim))
        print("     Norm   word     emb: %s"%(self.norm_word_emb))
        print("     Norm   char     emb: %s"%(self.norm_char_emb))
        print("     Train  file: %s"%(self.train_file))
        print("     Dev    file: %s"%(self.dev_file))
        print("     Test   file: %s"%(self.test_file))
        print("     OOV    file: %s"%(self.oov_file))
        print("     Train instance number: %s"%(len(self.train_text)))
        print("     Dev   instance number: %s"%(len(self.dev_text)))
        print("     Test  instance number: %s"%(len(self.test_text)))
        print("     OOV   instance number: %s"%(len(self.oov_text)))

        print(" "+"++"*20)
        print(" Model Network:")
        print("     Model        use_crf: %s"%(self.use_crf))
        print("     Model word extractor: %s"%(self.word_feature_extractor))
        print("     Model       use_char: %s"%(self.use_char))
        print("     Model      fine_tune: %s"%(self.fine_tune))
        print("     Model elmo_fine_tune: %s"%(self.elmo_fine_tune))
        if self.use_char:
            print("     Model char extractor: %s"%(self.char_feature_extractor))
            print("     Model char_hidden_dim: %s"%(self.char_hidden_dim))
        if self.feature:
            for idx in range(self.feature_num):
                print("     Feature name: %s"%(self.feature_names[idx]))
                print("     Feature dim: %s"%(self.feature_embed_dim[idx]))
        if self.out_dict:
            print("     use out dict.")
            if self.hyperlstm:
                print("     use hyperlstm.")
        print(" "+"++"*20)
        print(" Training:")
        print("     Optimizer: %s"%(self.optimizer))
        print("     Iteration: %s"%(self.iter))
        print("     BatchSize: %s"%(self.batch_size))
        print("     Average  batch   loss: %s"%(self.average_loss))

        print(" "+"++"*20)
        print(" Hyperparameters:")

        print("     Hyper              lr: %s"%(self.lr))
        print("     Hyper        weight_decay: %s"%(self.weight_decay))
        print("     Hyper      hidden_dim: %s"%(self.hidden_dim))
        print("     Hyper         dropout: %s"%(self.dropout))
        print("     Hyper      lstm_layer: %s"%(self.lstm_layer))
        print("     Hyper          bilstm: %s"%(self.bilstm))
        print("     Hyper             GPU: %s"%(self.use_cuda))
        print("     Hyper      Elmo concate embed: %s"%(self.use_elmo))
        print("     Hyper      char concate word: %s"%(self.use_char))
        print("     Hyper      char attention word: %s"%(self.attention))
        print("     Hyper      mask entity: %s"%(self.entity_mask))
        print("     Hyper      mask percent: %s"%(self.mask_percent))
        print("DATA SUMMARY END.")
        print("++"*50)
        sys.stdout.flush()


def build_pretrain_embedding(embedding_save, embedding_path, word_alphabet, emb_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_save is not None and os.path.exists(embedding_save):
        pretrain_emb = pickle.load(open(embedding_save, 'rb'))
        embedd_dim = pretrain_emb.shape[1]
        return pretrain_emb, embedd_dim
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.zeros([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        #elif word.lower() in embedd_dict:
        #    if norm:
        #        pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
        #    else:
        #        pretrain_emb[index, :] = embedd_dict[word.lower()]
        #    case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    pickle.dump(pretrain_emb, open(embedding_save, 'wb'))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        file.readline()   #中文词向量先读一行
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 != len(tokens):
                    continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def str2bool(string):
    if string == "True" or string == "true" or string == "TRUE":
        return True
    else:
        return False


def statistic_dataset_entity(path):
    LOC = MISC = ORG = PER = 0
    entity = None
    with open(path, 'r') as fin:
        for line in fin:
            if line == '\n':
                continue
            else:
                info = line.split()
                label = info[3]
                if "B-" in label:
                    if entity is not None:
                        if entity == "MISC":
                            MISC += 1
                        elif entity == "LOC":
                            LOC += 1
                        elif entity == "PER":
                            PER += 1
                        elif entity == "ORG":
                            ORG += 1
                        else:
                            print("entity Error.")
                        entity = label.split('-')[1]
                    else:
                        entity = label.split('-')[1]
                elif "I-" in label:
                    if entity is not None:
                        continue
                    else:
                        print(line,label,entity)
                        print("BIO Error.")
                elif "O" in label:
                    if entity is not None:
                        if entity == "MISC":
                            MISC += 1
                        elif entity == "LOC":
                            LOC += 1
                        elif entity == "PER":
                            PER += 1
                        elif entity == "ORG":
                            ORG += 1
                        else:
                            print("entity Error.")
                        entity = None
                    else:
                        continue
                else:
                    print("label Error.")
    print("LOC={}\tMISC={}\tORG={}\tPER={}\t".format(LOC, MISC, ORG, PER))


# if __name__ == "__main__":
#     print("train dataset:\t",end='')
#     statistic_dataset_entity("../data/conll2003/train.txt")
#     print("dev dataset:\t",end='')
#     statistic_dataset_entity("../data/conll2003/dev.txt")
#     print("test dataset:\t",end='')
#     statistic_dataset_entity("../data/conll2003/test.txt")

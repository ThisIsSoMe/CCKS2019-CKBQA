import torch
import jieba
import json
from torch.nn.utils.rnn import pad_sequence

# 加载词向量
def load_embed(seq_list,word_vector_in=None,d_vector=300,PAD='<PAD>',UNK='<UNK>'):
    words_dict={ PAD:0, UNK:1 }
    vector_list=[]
    if word_vector_in:
        with open(word_vector_in,'r') as f:
            for line in f:
                
                items=line.strip().split()
                char=items[0]
                #set_trace()
                if len(char)==1:
                    vector=[float(i) for i in items[1:]]
                    words_dict[char]=len(words_dict)
                    vector_list.append(vector)
    print("char from embed file:%d\n"%len(words_dict))
    for one_q in seq_list:
        for word in one_q:
            if word not in words_dict:
                words_dict[word]=len(words_dict)
    word_vector=(torch.rand(len(words_dict),d_vector)-0.5)/2 # [-0.25,0.25]的均匀分布
    if word_vector_in:
        word_vector[2:2+len(vector_list)]=torch.Tensor(vector_list)
    return words_dict,word_vector

class Corpus:
    def __init__(self, PAD='<PAD>', UNK='<UNK>', SOS='<SOS>', EOS='<EOS>', word_max_len=10):
        self.PAD = PAD
        self.UNK = UNK
        self.SOS = SOS
        self.EOS = EOS
        self.CHARS = [self.PAD, self.UNK, self.SOS, self.EOS]
        self.words_dict = {}
        self.word_max_len = word_max_len # 每个词中最多的字数
        for char in self.CHARS:
            self.words_dict[char] = len(self.words_dict)
        self.char_dict = {}
        for char in self.CHARS:
            self.char_dict[char] = len(self.char_dict)

    def load_embed(self, fn_embed):
        print("start load word embedding")
        vector_list=[]
        with open(fn_embed,'r') as f:
            index = -1
            for line in f:
                index += 1
                line = line.strip().split()
                if index == 0:
                    items = line
                    _ , dim = items[0], int(items[1])
                else:
                    items = line
                    word = items[0]
                    vector = [float(i) for i in items[1:]]
                    if word not in self.words_dict:
                        self.words_dict[word] = len(self.words_dict)
                        vector_list.append(vector)
        word_vector=(torch.rand((len(self.words_dict), dim))-0.5)/2 # [-0.25,0.25]的均匀分布
        word_vector[len(self.CHARS):len(self.CHARS)+len(vector_list)]=torch.Tensor(vector_list)
        return self.words_dict, word_vector

    def load_data(self, fn_data, mode):
        # mode: train, valid, test
        with open(fn_data, 'r')as f:
            data = json.load(f)
        if mode == 'train' or mode == 'valid':
            questions, golds, negs = data['questions'], data['golds'], data['negs']
            return (questions, golds, negs)
        elif mode == 'test':
            questions, cands = data['questions'], data['cands']
            return (questions, cands)
    
    def len_char_dict(self):
        return len(self.char_dict)
    
    def dump_vocab(self, path, mode='word'):
        if mode == 'word':
            with open(path, 'w')as f:
                json.dump(self.words_dict, f, ensure_ascii=False)
        elif mode == 'char':
            with open(path, 'w')as f:
                json.dump(self.char_dict, f, ensure_ascii=False)
        else:
            print("Mode error! Please check the mode")

    def numericalize(self, sentences, mode, words_dict=None, char_dict=None, state='train'):
        # mode: word, char, word_char

        if not words_dict:
            words_dict = self.words_dict
        if not char_dict:
            char_dict = self.char_dict

        PID = words_dict.get(self.PAD)
        UID = words_dict.get(self.UNK)
        SID = words_dict.get(self.SOS)
        EID = words_dict.get(self.EOS)

        if mode == 'word':
            sents = []
            for sentence in sentences:
                sentence = [word for word in jieba.cut(sentence)]
                sent = [words_dict.get(word, UID) for word in sentence]
                sents.append(sent)

            max_seq_len = 0
            for s in sents:
                max_seq_len = max(max_seq_len, len(s))
            for i, s in enumerate(sents):
                sents[i] = s + [PID]*(max_seq_len - len(s))
            return torch.LongTensor(sents)

        elif mode == 'char':
            sents = []
            for sentence in sentences:
                sent = [words_dict.get(char, UID) for char in sentence]
                sents.append(sent)
            max_seq_len = 0
            for s in sents:
                max_seq_len = max(max_seq_len, len(s))
            for i, s in enumerate(sents):
                sents[i] = s + [PID]*(max_seq_len - len(s))
            return torch.LongTensor(sents)

        elif mode == 'word_char':
            sents = []
            sents_char = []
            for sentence in sentences:
                words = [word for word in jieba.cut(sentence)]
                sent = [words_dict.get(word, UID) for word in words]

                if state != 'test':
                    for char in sentence:
                        if char not in char_dict:
                            char_dict[char] = len(char_dict)

                chars = [[char_dict.get(char, UID) for char in word[:self.word_max_len]] + [PID]*(self.word_max_len - len(word)) for word in words]
                sents.append(sent)
                sents_char.append(chars)
                
            # 对词进行pad
            max_seq_len = 0
            for s in sents:
                max_seq_len = max(max_seq_len, len(s))
            for i, s in enumerate(sents):
                sents[i] = s + [PID]*(max_seq_len - len(s))
            # 对字进行pad
            if sents_char:
                sents_char = pad_sequence([torch.LongTensor(line) for line in sents_char], True)
            return (torch.LongTensor(sents), torch.LongTensor(sents_char))
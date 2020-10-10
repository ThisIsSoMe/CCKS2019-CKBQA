import torch
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

    def simplize(self, seq):
        chars = ['<','>']
        new_seq = []
        for word in seq:
            if word not in chars:
                new_seq.append(word)
        return new_seq

    def numericalize(self, sequences):
        subwords, lens = [], []
        sequences = [([self.bos] if self.bos else []) + list(sequence) +
                     ([self.sep] if self.sep else [])
                     for sequence in sequences]

        origin_len = len(sequences)

        for one_sequence in sequences:
            sequence = one_sequence
            sequence = [self.transform(token) for token in sequence]
            if [] in sequence:
                sequence.remove([])
            sequence = [piece if piece else self.transform(self.pad)
                        for piece in sequence]
            subwords.append(sum(sequence, []))
            lens.append(torch.tensor([len(piece) for piece in sequence]))
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).ge(0) for pieces in subwords]

        assert origin_len == len(lens)
        return (subwords, lens, mask)

class BertCharField(Field):
    
    def numericalize(self, sequences):
        tmp = sequences

        sequences = [ [self.bos] + self.tokenizer.tokenize(sequence) for sequence in sequences]
        sequences = [self.tokenizer.convert_tokens_to_ids(sequence) for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]
        mask = [torch.ones(len(sequence)) for sequence in sequences]
        return (sequences, mask)

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
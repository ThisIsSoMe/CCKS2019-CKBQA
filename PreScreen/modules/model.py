import sys 
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter, ParameterList
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pad_sequence
from modules.charlstm import CHAR_LSTM

class BiLSTM_Encoding_Comparing(nn.Module):
    def __init__(self, args, word_emb):
        super(BiLSTM_Encoding_Comparing, self).__init__()
        # parameters
        self.hidden_size = args.hidden_size
        self.question_layers = args.question_layers
        self.path_layers = args.path_layers
        self.dropout_rate = args.dropout_rate
        self.fine_tune = args.fine_tune
        self.dict_size = word_emb.shape[0]
        self.word_embedding_dim = word_emb.shape[1]
        # word embedding layer
        self.word_embedding = nn.Embedding(self.dict_size, self.word_embedding_dim)
        self.word_embedding.weight = nn.Parameter(word_emb)
        self.word_embedding.weight.requires_grad = self.fine_tune # whether fix the embedding matrix
        # rnn layer
        self.question_bilstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, num_layers=self.question_layers,bidirectional=True, batch_first=True)
        self.path_bilstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, num_layers=self.path_layers,bidirectional=True, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # similarity calculation
        self.cos = nn.CosineSimilarity(1)
    
    def forward(self, question, pos, negs):
        '''
        question: batch_size,max_seq_len
        pos: batch_size,max_seq_len
        negs: batch_size,neg_size,max_seq_len 
        '''
        # question encoding
        question_encode = self.question_encoder(question) # [batch_size, hidden_size]
        
        # pos encoding
        pos_encode = self.path_encoder(pos) # [batch_size, hidden_size]

        # neg encoding
        batch_size, neg_size, _ = negs.shape
        negs = negs.reshape(batch_size*neg_size, -1) # [batch_size*neg_size, hidden_size]
        negs_encode = self.path_encoder(negs)

        # reshape
        hidden_size = question_encode.shape[-1]
        
        question_encode_expand = question_encode.unsqueeze(1).expand(batch_size,neg_size,hidden_size).reshape(-1, hidden_size)
        assert question_encode_expand.shape == negs_encode.shape
        # pos_score = self.cos(question_encode_expand, pos_encode_expand) # [batch_size*neg_size, 1]
        pos_score = self.cos(question_encode, pos_encode) # [batch_size]
        neg_score = self.cos(question_encode_expand, negs_encode) # [batch_size*neg_size, 1]
        
        pos_score = pos_score.unsqueeze(1) # [batch_size, 1]
        neg_score = neg_score.reshape(batch_size, neg_size) # [batch_size, neg_size]
        return pos_score, neg_score


    def question_encoder(self, question):
        question_embed = self.word_embedding(question)
        question_mask = question.gt(0)
        question_lens = torch.sum(question_mask, dim=-1)
        sorted_lens, sorted_idx = torch.sort(question_lens, descending=True)
        question_embed = question_embed[sorted_idx, :sorted_lens[0]]

        lstm_input = pack(question_embed, sorted_lens, batch_first=True)
        question_out, (ht, ct) = self.question_bilstm(lstm_input)
        
        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        question_encode = ht
        return question_encode
    
    def path_encoder(self, path):
        path_embed = self.word_embedding(path)
        path_mask = path.gt(0)
        path_lens = torch.sum(path_mask, dim=-1)
        sorted_lens, sorted_idx = torch.sort(path_lens, descending=True)
        path_embed = path_embed[sorted_idx, :sorted_lens[0]]

        lstm_input = pack(path_embed, sorted_lens, batch_first=True)
        path_out , (ht, ct) = self.path_bilstm(lstm_input)

        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        path_encode = ht
        return path_encode
    
    def cal_score(self, question, paths):
        '''
        q: [1, max_seq_len]
        paths:[case,max_seq_len]
        '''
        question_encode = self.question_encoder(question)

        paths_encode = self.path_encoder(paths) # [batch_size, hidden_size]
        
        question_encode_expand = question_encode.expand_as(paths_encode)
        
        score = self.cos(question_encode_expand, paths_encode)
        return score

class Char_BiLSTM_Encoding_Comparing(nn.Module):
    def __init__(self, args, word_emb):
        super(Char_BiLSTM_Encoding_Comparing, self).__init__()
        self.args = args
        # parameters
        self.hidden_size = args.hidden_size
        self.question_layers = args.question_layers
        self.path_layers = args.path_layers
        self.dropout_rate = args.dropout_rate
        self.fine_tune = args.fine_tune
        self.dict_size = word_emb.shape[0]
        self.word_embedding_dim = word_emb.shape[1]
        # word embedding layer
        self.word_embedding = nn.Embedding(self.dict_size, self.word_embedding_dim)
        self.word_embedding.weight = nn.Parameter(word_emb)
        self.word_embedding.weight.requires_grad = self.fine_tune # whether fix the embedding matrix
        # rnn layer
        # char lstm
        self.char_lstm = CHAR_LSTM(args.n_chars, args.dim_char, args.char_out) # default:batch first
        self.question_bilstm = nn.LSTM(self.word_embedding_dim+args.char_out, self.hidden_size, num_layers=self.question_layers,bidirectional=True, batch_first=True)
        self.path_bilstm = nn.LSTM(self.word_embedding_dim+args.char_out, self.hidden_size, num_layers=self.path_layers,bidirectional=True, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # similarity calculation
        self.cos = nn.CosineSimilarity(1)
    
    def forward(self, question, pos, negs):
        '''
        question: batch_size,max_seq_len
        pos: batch_size,max_seq_len
        negs: batch_size,neg_size,max_seq_len
        '''
        # question encoding
        
        question_encode = self.question_encoder(question) # [batch_size, hidden_size]
        
        # pos encoding
        pos_encode = self.path_encoder(pos) # [batch_size, hidden_size]

        # neg encoding
        negs_word, negs_char = negs
        batch_size, neg_size, word_len, char_len = negs_char.shape
        negs_word = negs_word.reshape(batch_size*neg_size, -1) # [batch_size*neg_size, hidden_size]
        negs_char = negs_char.reshape(batch_size*neg_size, word_len, -1) # [batch_size*neg_size, hidden_size]
        negs_encode = self.path_encoder((negs_word, negs_char))

        # reshape
        hidden_size = question_encode.shape[-1]
        
        question_encode_expand = question_encode.unsqueeze(1).expand(batch_size,neg_size,hidden_size).reshape(-1, hidden_size)
        assert question_encode_expand.shape == negs_encode.shape
        # pos_score = self.cos(question_encode_expand, pos_encode_expand) # [batch_size*neg_size, 1]
        pos_score = self.cos(question_encode, pos_encode) # [batch_size]
        neg_score = self.cos(question_encode_expand, negs_encode) # [batch_size*neg_size, 1]
        
        pos_score = pos_score.unsqueeze(1) # [batch_size, 1]
        neg_score = neg_score.reshape(batch_size, neg_size) # [batch_size, neg_size]
        return pos_score, neg_score


    def question_encoder(self, input):
        if not self.args.no_cuda:
            input = (t.cuda() for t in input)

        question, question_chars = input

        # word level
        question_mask = question.gt(0)
        question_lens = torch.sum(question_mask, dim=-1)

        max_len = torch.max(question_lens)
        question_embed = self.word_embedding(question[:, :max_len])

        # char level
        char_x = self.char_lstm(question_chars[question_mask])
        char_x = pad_sequence(torch.split(char_x, question_lens.tolist()), True)
        question_embed = torch.cat((question_embed,char_x),-1)

        question_embed = self.dropout(question_embed)
        
        sorted_lens, sorted_idx = torch.sort(question_lens, descending=True)
        question_embed = question_embed[sorted_idx, :sorted_lens[0]]

        lstm_input = pack(question_embed, sorted_lens, batch_first=True)
        question_out, (ht, ct) = self.question_bilstm(lstm_input)
        
        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        question_encode = ht
        return question_encode
    
    def path_encoder(self, input):
        if not self.args.no_cuda:
            input = (t.cuda() for t in input)
        path, path_chars = input
        
        path_mask = path.gt(0)
        path_lens = torch.sum(path_mask, dim=-1)

        max_len = torch.max(path_lens)
        path_embed = self.word_embedding(path[:, :max_len])
        # char level
        char_x = self.char_lstm(path_chars[path_mask])
        char_x = pad_sequence(torch.split(char_x, path_lens.tolist()), True)
        path_embed = torch.cat((path_embed,char_x),-1)

        path_embed = self.dropout(path_embed)
        sorted_lens, sorted_idx = torch.sort(path_lens, descending=True)
        path_embed = path_embed[sorted_idx, :sorted_lens[0]]

        lstm_input = pack(path_embed, sorted_lens, batch_first=True)
        path_out , (ht, ct) = self.path_bilstm(lstm_input)

        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        path_encode = ht
        return path_encode
    
    def cal_score(self, question, paths):
        '''
        q: [1, max_seq_len]
        paths: [case,max_seq_len]
        '''
        # from pdb import set_trace
        # set_trace()
        question_encode = self.question_encoder(question)

        paths_encode = self.path_encoder(paths) # [batch_size, hidden_size]
        
        question_encode_expand = question_encode.expand_as(paths_encode)

        score = self.cos(question_encode_expand, paths_encode)
        return score

class Syntax_BiLSTM_Encoding_Comparing(nn.Module):
    def __init__(self, args, word_emb):
        super(Syntax_BiLSTM_Encoding_Comparing, self).__init__()
        self.args = args
        # parameters
        self.hidden_size = args.hidden_size
        self.question_layers = args.question_layers
        self.path_layers = args.path_layers
        self.dropout_rate = args.dropout_rate
        self.fine_tune = args.fine_tune
        self.dict_size = word_emb.shape[0]
        self.word_embedding_dim = word_emb.shape[1]
        # word embedding layer
        self.word_embedding = nn.Embedding(self.dict_size, self.word_embedding_dim)
        self.word_embedding.weight = nn.Parameter(word_emb)
        self.word_embedding.weight.requires_grad = self.fine_tune # whether fix the embedding matrix
        # rnn layer
        # char lstm
        self.char_lstm = CHAR_LSTM(args.n_chars, args.dim_char, args.char_out) # default:batch first
        self.question_bilstm = nn.LSTM(self.word_embedding_dim+args.char_out+100, self.hidden_size, num_layers=self.question_layers,bidirectional=True, batch_first=True)
        self.path_bilstm = nn.LSTM(self.word_embedding_dim+args.char_out, self.hidden_size, num_layers=self.path_layers,bidirectional=True, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        # syntax layer
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scalar_parameters = ParameterList([Parameter(torch.FloatTensor([0]),
                   requires_grad=True) for i in range(args.num_syntax_layers)])
        self.compress = nn.Linear(args.dim_syntax, 100)

        # similarity calculation
        self.cos = nn.CosineSimilarity(1)
    
    def forward(self, question, pos, negs, syntax):
        '''
        question: batch_size,max_seq_len
        pos: batch_size,max_seq_len
        negs: batch_size,neg_size,max_seq_len
        '''
        # question encoding
        
        question_encode = self.question_encoder(question, syntax) # [batch_size, hidden_size]
        
        # pos encoding
        pos_encode = self.path_encoder(pos) # [batch_size, hidden_size]

        # neg encoding
        negs_word, negs_char = negs
        batch_size, neg_size, word_len, char_len = negs_char.shape
        negs_word = negs_word.reshape(batch_size*neg_size, -1) # [batch_size*neg_size, hidden_size]
        negs_char = negs_char.reshape(batch_size*neg_size, word_len, -1) # [batch_size*neg_size, hidden_size]
        negs_encode = self.path_encoder((negs_word, negs_char))

        # reshape
        hidden_size = question_encode.shape[-1]
        
        question_encode_expand = question_encode.unsqueeze(1).expand(batch_size,neg_size,hidden_size).reshape(-1, hidden_size)
        assert question_encode_expand.shape == negs_encode.shape
        # pos_score = self.cos(question_encode_expand, pos_encode_expand) # [batch_size*neg_size, 1]
        pos_score = self.cos(question_encode, pos_encode) # [batch_size]
        neg_score = self.cos(question_encode_expand, negs_encode) # [batch_size*neg_size, 1]
        
        pos_score = pos_score.unsqueeze(1) # [batch_size, 1]
        neg_score = neg_score.reshape(batch_size, neg_size) # [batch_size, neg_size]
        return pos_score, neg_score


    def question_encoder(self, input, syntax_input):
        if not self.args.no_cuda:
            input = (t.cuda() for t in input)
            if self.args.use_syntax:
                syntax_input = syntax_input.cuda() # [B, L, layers, D]

        question, question_chars = input
        # syntax embeddding
        normed_weights = F.softmax(torch.cat([parameter for parameter
	                                                        in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        pieces = []
        for i in range(self.args.num_syntax_layers):
            w = normed_weights[i]
            vec = syntax_input[:, :, i, :]
            pieces.append(w*vec)
        syntax_embed = self.gamma * sum(pieces) # [B,L,D]
        syntax_embed = self.compress(syntax_embed)
        # from pdb import set_trace
        # set_trace()
        # word level
        question_mask = question.gt(0)
        question_lens = torch.sum(question_mask, dim=-1)

        max_len = torch.max(question_lens)
        question_embed = self.word_embedding(question[:, :max_len])

        # char level
        char_x = self.char_lstm(question_chars[question_mask])
        char_x = pad_sequence(torch.split(char_x, question_lens.tolist()), True)
        # cat char embed
        question_embed = torch.cat((question_embed,char_x),-1)
        # cat syntax embed
        question_embed = torch.cat((question_embed,syntax_embed),-1)

        question_embed = self.dropout(question_embed)
        
        sorted_lens, sorted_idx = torch.sort(question_lens, descending=True)
        question_embed = question_embed[sorted_idx, :sorted_lens[0]] # [B,L,D]
        
        lstm_input = pack(question_embed, sorted_lens, batch_first=True)
        question_out, (ht, ct) = self.question_bilstm(lstm_input)
        
        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        question_encode = ht
        return question_encode
    
    def path_encoder(self, input):
        if not self.args.no_cuda:
            input = (t.cuda() for t in input)
        path, path_chars = input
        
        path_mask = path.gt(0)
        path_lens = torch.sum(path_mask, dim=-1)

        max_len = torch.max(path_lens)
        path_embed = self.word_embedding(path[:, :max_len])
        # char level
        char_x = self.char_lstm(path_chars[path_mask])
        char_x = pad_sequence(torch.split(char_x, path_lens.tolist()), True)
        path_embed = torch.cat((path_embed,char_x),-1)

        path_embed = self.dropout(path_embed)
        sorted_lens, sorted_idx = torch.sort(path_lens, descending=True)
        path_embed = path_embed[sorted_idx, :sorted_lens[0]]

        lstm_input = pack(path_embed, sorted_lens, batch_first=True)
        path_out , (ht, ct) = self.path_bilstm(lstm_input)

        _, reverse_idx = torch.sort(sorted_idx)
        ht = ht[:, reverse_idx, :]
        ht = torch.cat((ht[-1],ht[-2]),-1)
        path_encode = ht
        return path_encode
    
    def cal_score(self, question, paths, syntax_embed):
        '''
        q: [1, max_seq_len]
        paths: [case,max_seq_len]
        '''
        # from pdb import set_trace
        # set_trace()
        question_encode = self.question_encoder(question, syntax_embed)

        paths_encode = self.path_encoder(paths) # [batch_size, hidden_size]
        
        question_encode_expand = question_encode.expand_as(paths_encode)

        score = self.cos(question_encode_expand, paths_encode)
        return score

class HR_BiLSTM(nn.Module):
    def __init__(self, dropout_rate, hidden_size, word_emb, rela_emb, requires_grad=True):
        super(HR_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layers = 1
        self.dropout_rate = dropout_rate
        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = requires_grad # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(torch.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = requires_grad # fix the embedding matrix
        # LSTM layer
        self.bilstm_1 = nn.LSTM(word_emb.shape[1], hidden_size, num_layers=self.nb_layers, bidirectional=True)
        self.bilstm_2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=self.nb_layers, bidirectional=True)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.cos = nn.CosineSimilarity(1)
        
    def forward(self, question, rela_relation, word_relation):
        question = torch.transpose(question, 0, 1)
        rela_relation = torch.transpose(rela_relation, 0, 1)
        word_relation = torch.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        #print('question_emb.shape', question.shape)
        rela_relation = self.rela_embedding(rela_relation)
        #print('rela_relation_emb.shape', rela_relation.shape)
        word_relation = self.word_embedding(word_relation)
        #print('word_relation_emb.shape', word_relation.shape)
        #print()

        question = self.dropout(question)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.dropout(word_relation)

#        self.bilstm_1.flatten_parameters()
        question_out_1, question_hidden = self.bilstm_1(question)
        question_out_1 = self.dropout(question_out_1)
        #print('question_out_1.shape', question_out_1.shape)
#        self.bilstm_2.flatten_parameters()
        question_out_2, _ = self.bilstm_2(question_out_1)
        question_out_2 = self.dropout(question_out_2)
        #print('question_out_2.shape', question_out_2.shape)
        
        # 1st way of Hierarchical Residual Matching
        q12 = question_out_1 + question_out_2
        q12 = q12.permute(1, 2, 0)
        #print('q12.shape', q12.shape)
        question_representation = nn.MaxPool1d(q12.shape[2])(q12)
        question_representation = question_representation.squeeze(2)
        # 2nd way of Hierarchical Residual Matching
        #q1_max = nn.MaxPool1d(question_out_1.shape[2])(question_out_1)
        #q2_max = nn.MaxPool1d(question_out_2.shape[2])(question_out_2)
        #question_representation = q1_max + q2_max
        #print('question_representation.shape', question_representation.shape) 

        #print()
#        self.bilstm_1.flatten_parameters()
        word_relation_out, word_relation_hidden = self.bilstm_1(word_relation)
        word_relation_out = self.dropout(word_relation_out)
        #print('word_relation_out.shape', word_relation_out.shape)
#        self.bilstm_1.flatten_parameters()
        rela_relation_out, rela_relation_hidden = self.bilstm_1(rela_relation, word_relation_hidden)
        rela_relation_out = self.dropout(rela_relation_out)
        #print('rela_relation_out.shape', rela_relation_out.shape)
        r = torch.cat([rela_relation_out, word_relation_out], 0)
        r = r.permute(1, 2, 0)
        #print('r.shape', r.shape)
        relation_representation = nn.MaxPool1d(r.shape[2])(r)
        relation_representation = relation_representation.squeeze(2)
        #print('relation_representation.shape', relation_representation.shape)

        score = self.cos(question_representation, relation_representation)
        #print('score.shape', score.shape)
        return score
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.ques_embedding = nn.Embedding(args.ques_embedding.shape[0], args.ques_embedding.shape[1])
        self.ques_embedding.weight.requires_grad = False
        self.ques_embedding.weight = nn.Parameter(torch.from_numpy(args.ques_embedding).float())
        self.rela_text_embedding = nn.Embedding(args.rela_text_embedding.shape[0], args.rela_text_embedding.shape[1])
        self.rela_text_embedding.weight.requires_grad = False
        self.rela_text_embedding.weight = nn.Parameter(torch.from_numpy(args.rela_text_embedding).float())
        self.rela_embedding = nn.Embedding(args.rela_vocab_size, args.rela_text_embedding.shape[1])
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size,
                          num_layers=args.num_layers, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=args.hidden_size*2, hidden_size=args.hidden_size,
                          num_layers=args.num_layers, batch_first=False,
                          dropout=args.dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args
        self.cos = nn.CosineSimilarity(dim=1)
        self.tanh = nn.Tanh()
        return

    def forward(self, ques_x, rela_text_x, rela_x):
        ques_x = torch.transpose(ques_x, 0, 1)
        rela_text_x = torch.transpose(rela_text_x, 0, 1)
        rela_x = torch.transpose(rela_x, 0, 1)

        ques_x = self.ques_embedding(ques_x)
        rela_text_x = self.rela_text_embedding(rela_text_x)
        rela_x = self.rela_embedding(rela_x)

        ques_hs1, hidden_state = self.encode(ques_x)
        rela_hs, hidden_state = self.encode(rela_x, hidden_state)
        rela_text_hs, hidden_state = self.encode(rela_text_x, hidden_state)

        h_0 = Variable(torch.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        c_0 = Variable(torch.zeros([self.args.num_layers*2, len(ques_x[0]), self.args.hidden_size])).cuda()
        ques_hs2, _ = self.rnn2(ques_hs1, (h_0, c_0)) 

        ques_hs = ques_hs1 + ques_hs2
        ques_hs = ques_hs.permute(1, 2, 0)
        ques_h = F.max_pool1d(ques_hs, kernel_size=ques_hs.shape[2], stride=None)
        rela_hs = torch.cat([rela_hs, rela_text_hs], 0)
        rela_hs = rela_hs.permute(1, 2, 0)
        rela_h = F.max_pool1d(rela_hs, kernel_size=rela_hs.shape[2], stride=None)

        ques_h = ques_h.squeeze(2)
        rela_h = rela_h.squeeze(2)

        output = self.cos(ques_h, rela_h)
        return output

    def encode(self, input, hidden_state=None, return_sequence=True):
        if hidden_state==None:
            h_0 = Variable(torch.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).cuda()
            c_0 = Variable(torch.zeros([self.args.num_layers*2, len(input[0]), self.args.hidden_size])).cuda()
        else:
            h_0, c_0 = hidden_state
        h_input = h_0
        c_input = c_0
        outputs, (h_output, c_output) = self.rnn(input, (h_0, c_0))
        if return_sequence == False:
            return outputs[-1], (h_output, c_output)
        else:
            return outputs, (h_output, c_output)

class ABWIM(nn.Module):
    def __init__(self, dropout_rate, hidden_size, word_emb, rela_emb, q_len, r_len):
        super(ABWIM, self).__init__()
        self.hidden_size = hidden_size
        self.nb_layers = 1
        self.nb_filters = 100
        self.dropout = nn.Dropout(dropout_rate)
        # Word Embedding layer
        self.word_embedding = nn.Embedding(word_emb.shape[0], word_emb.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_emb).float())
        self.word_embedding.weight.requires_grad = False # fix the embedding matrix
        # Rela Embedding layer
        self.rela_embedding = nn.Embedding(rela_emb.shape[0], rela_emb.shape[1])
        self.rela_embedding.weight = nn.Parameter(torch.from_numpy(rela_emb).float())
        self.rela_embedding.weight.requires_grad = False # fix the embedding matrix
        # LSTM layer
        self.bilstm = nn.LSTM(word_emb.shape[1], 
                              hidden_size, 
                              num_layers=self.nb_layers, 
                              bidirectional=True)
        # Attention
        self.W = nn.Parameter(torch.rand((hidden_size*2, hidden_size*2))).cuda()
        # CNN layer
        self.cnn_1 = nn.Conv1d(hidden_size*4, self.nb_filters, 1)
        self.cnn_2 = nn.Conv1d(hidden_size*4, self.nb_filters, 3)
        self.cnn_3 = nn.Conv1d(hidden_size*4, self.nb_filters, 5)
        self.activation = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(q_len)
        self.maxpool_2 = nn.MaxPool1d(q_len-2)
        self.maxpool_3 = nn.MaxPool1d(q_len-4)
        self.linear = nn.Linear(self.nb_filters, 1, bias=False)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda())
   
    def forward(self, question, rela_relation, word_relation):
        question = torch.transpose(question, 0, 1)
        rela_relation = torch.transpose(rela_relation, 0, 1)
        word_relation = torch.transpose(word_relation, 0, 1)

        question = self.word_embedding(question)
        question = self.dropout(question)
        rela_relation = self.rela_embedding(rela_relation)
        rela_relation = self.dropout(rela_relation)
        word_relation = self.word_embedding(word_relation)
        word_relation = self.dropout(word_relation)

#        self.bilstm.flatten_parameters()
        question_out, _ = self.bilstm(question)
        question_out = question_out.permute(1,2,0)
        question_out = self.dropout(question_out)
        word_relation_out, word_relation_hidden = self.bilstm(word_relation)
        rela_relation_out, _ = self.bilstm(rela_relation, word_relation_hidden)
        word_relation_out = self.dropout(word_relation_out)
        rela_relation_out = self.dropout(rela_relation_out)
        relation = torch.cat([rela_relation_out, word_relation_out], 0)
        relation = relation.permute(1,0,2)

        # attention layer
        energy = torch.matmul(relation, self.W)
        energy = torch.matmul(energy, question_out)
        alpha = F.softmax(energy, dim=-1)
        alpha = alpha.unsqueeze(3)
        relation = relation.unsqueeze(2)
        atten_relation = alpha * relation
        atten_relation = torch.sum(atten_relation, 1)
        atten_relation = atten_relation.permute(0, 2, 1)
        M = torch.cat((question_out, atten_relation), 1)
        h1 = self.maxpool_1(self.activation(self.cnn_1(M)))
        h1 = self.dropout(h1)
        h2 = self.maxpool_2(self.activation(self.cnn_2(M)))
        h2 = self.dropout(h2)
        h3 = self.maxpool_3(self.activation(self.cnn_3(M)))
        h3 = self.dropout(h3)
        h = torch.cat((h1, h2, h3),2)
        h = torch.max(h, 2)[0]
        score = self.linear(h).squeeze()
        return score
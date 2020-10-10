from __future__ import absolute_import, division, print_function
import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert import *
#from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

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
            #import pdb;pdb.set_trace()
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1","2","3"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_syntax = line[1] 
            label = line[2] 
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, text_syntax=text_syntax))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    #import pdb;pdb.set_trace()
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens) 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
 
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def convert_syntax_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, syntax_dict, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_syntax = example.text_syntax.split(' ')
        syntax_labels = tokens_syntax
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        syntax_labels = ["<CLS>"] + syntax_labels + ["<SEP>"]
        syntax_ids = [syntax_dict.get(syn,0) for syn in syntax_labels]
        
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        syntax_ids += padding
 
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        print(len(syntax_ids),max_seq_length,syntax_ids,input_ids,tokens,example.text_syntax)
        assert len(syntax_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures_syntax(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              syntax_ids=syntax_ids))
    return features


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mrpc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

def main():
    bert_path="../../data/"
    parser = ArgumentParser(description = 'For KBQA')
    parser.add_argument("--data_dir",default='train_data_syntax_char',type=str)
    parser.add_argument("--bert_path",default=bert_path,type=str)
    parser.add_argument("--bert_model", default=bert_path+'bert-base-chinese.tar.gz', type=str)
    parser.add_argument("--bert_vocab", default=bert_path+'bert-base-chinese-vocab.txt', type=str)
    parser.add_argument("--task_name",default='mrpc',type=str,help="The name of the task to train.")
    parser.add_argument("--output_dir",default='saved_syntax_char',type=str)
    ## Other parameters
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",default=55,type=int)
    parser.add_argument("--do_train",default='true',help="Whether to run training.")
    parser.add_argument("--do_eval",default='true',help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",default=32,type=int,help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",default=32,type=int,help="Total batch size for eval.")
    parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",default=35,type=float,help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,)
    parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',type=int,default=42,help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    ## lstm parameters
    parser.add_argument("--use_syntax",action='store_true',help="Whether not to use CUDA when available")
    parser.add_argument("--bert_embedding_size",default=768,type=int)
    parser.add_argument("--hidden_dim",default=300,type=int)
    parser.add_argument("--lstm_layer",default=1,type=int)
    parser.add_argument("--bilstm",default='true')
    parser.add_argument("--len_syntax_dict",default=30,type=int)
    parser.add_argument("--syntax_dim",default=50,type=int)
    parser.add_argument("--num_labels",default=4,type=int)
    parser.add_argument("--dropout",default=0.5,type=float)
    args = parser.parse_args()
    
    label_list=['<PAD>','<CLS>','<SEP>','<ROOT>','ADV','AMOD','APP','AUX','BNF','CJT','CJTN','CJTN0','CJTN1','CJTN2','CJTN3','CJTN4','CJTN5','CJTN6','CJTN7','CJTN8','CJTN9','CND','COMP','DIR','DMOD','EXT','FOC','IO','LGS','LOC','MNR','NMOD','OBJ','OTHER','PRD','PRP','PRT','RELC','ROOT','SBJ','TMP','TPC','UNK','VOC','cCJTN']
    syntax_dict={}
    for item in label_list:
        syntax_dict[item]=len(syntax_dict)
    args.len_syntax_dict=len(syntax_dict)
    
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
        
    processors = {"mrpc": MrpcProcessor}
    output_modes = {"mrpc": "classification"}
    
    no_gpu=3
    n_gpu=1 
    device = torch.device("cuda", no_gpu)
    print("使用GPU%d"%no_gpu)
    print("                     ")
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        pass
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    # label数量
    num_labels = len(label_list)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)
    
    
    print("loaded tokenizer")
    
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs       

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = Bert_Classifier(args)
    print("loaded Bert model")
    model.to(device)
    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]        
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_eval:
        best_precision=0
        patience=10
        iters_left = patience

        eval_examples = processor.get_dev_examples(args.data_dir)
        
        if args.use_syntax:
            eval_features = convert_syntax_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, syntax_dict, output_mode)
        else:
            eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        if args.use_syntax:
            all_syntax_ids = torch.tensor([f.syntax_ids for f in eval_features], dtype=torch.long) 
        

        if output_mode == "classification":
            all_label_ids_eval = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        else:
            print("output mode error!!!")

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids_eval)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        eval_batch_num = len(eval_dataloader)

    if args.do_train:
        n_batch_correct = 0
        len_train_data = 0
        i_train_step = 0
        if args.use_syntax:
            train_features = convert_syntax_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, syntax_dict, output_mode)
        else:
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        if args.use_syntax:
            all_syntax_ids = torch.tensor([f.syntax_ids for f in train_features], dtype=torch.long) 
        if args.use_syntax:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_syntax_ids)
        else:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        train_batch_num=len(train_dataloader)     
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                i_train_step += 1
                batch = tuple(t.to(device) for t in batch)
                if args.use_syntax:
                    input_ids, input_mask, segment_ids, label_ids, syntax_ids = batch
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch
                if args.use_syntax:
                    logits = model(input_ids,input_mask,syntax_ids)
                else:
                    logits = model(input_ids,input_mask)
                
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                else:
                    print("output mode error!!!")
                loss.backward()
                n_batch_correct += torch.sum((torch.max(logits, 1)[1].data == label_ids.data))
                len_train_data += logits.size(0)               
                if i_train_step %train_batch_num ==0:
                    P_train = 1. * int(n_batch_correct)/len_train_data
                    print("                                             ")
                    print("-------------------------------------------------------------------")
                    print("epoch:%d\ttrain_Accuracy-----------------------%d/%d=%f\n"%(epoch,n_batch_correct.data,len_train_data,P_train))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:                   
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            model.to(device)
            if args.do_eval:
                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []
                n_dev_batch_correct = 0
                len_dev_data = 0
                i_dev_times = 0
                P_dev = 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    i_dev_times += 1
                    if args.use_syntax:
                        input_ids, input_mask, segment_ids, label_ids, syntax_ids = batch
                    else:
                        input_ids, input_mask, segment_ids, label_ids = batch
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    if args.use_syntax:
                        syntax_ids = syntax_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, input_mask)

                    # create eval loss and other metric required by the task
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    else:
                        print("output mode error!!!")
                    n_dev_batch_correct += torch.sum((torch.max(logits, 1)[1].data == label_ids.data))
                    len_dev_data += logits.size(0)  

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)
                
                P_dev = 1. * int(n_dev_batch_correct)/len_dev_data
                print()
                print("                                             ")
                print("-------------------------------------------------------------------")
                print("epoch:%d\tdev_Accuracy-----------------------%d/%d=%f\n"%(epoch,n_dev_batch_correct.data,len_dev_data,P_dev))
                if P_dev > best_precision:
                    best_precision = P_dev
                    iters_left = patience
                    if args.do_eval:
                        print("epoch %d saved\n"%epoch)
                        torch.save(model.state_dict(),args.output_dir+'/model_best.pkl')
                else:
                    iters_left-=1
                    if iters_left == 0:
                        break       
                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                if output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                else:
                    print("output mode error!!!")
                    
if __name__ == "__main__":
    main()

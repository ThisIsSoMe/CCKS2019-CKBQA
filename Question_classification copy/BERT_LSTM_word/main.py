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

def test(model, examples, bert_field, syn_field, dataloader, args, label_map,device):
    '''
    return:
    evaluate:(n_test_correct, n_test_data, 1.0*n_test_correct/n_test_data)
    result:[(gold label, predict label)]
    '''
    model.eval()
    test_examples = examples
    all_input_ids, all_input_lens, all_input_mask = bert_field.numericalize([example.text_a for example in test_examples])
    
    if args.use_syntax:
        all_syntax_ids = syn_field.numericalize([example.text_syntax for example in test_examples]) 
        all_syntax_embed = torch.load(args.syntax_embed_path%'test') 

    all_label_ids = torch.tensor([label_map[f.label] for f in test_examples], dtype=torch.long)

    if args.use_syntax:
        if args.syntax_hidden_embed:
            test_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_embed, hidden_embed=args.syntax_hidden_embed)
        else:
            test_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_ids, hidden_embed=args.syntax_hidden_embed)
    else:
        test_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids)
    
    test_batch_num = len(test_batches)

    n_test_correct = 0
    n_test_data = 0
    results = []
    for batch in test_batches:
        batch = tuple(t.to(device) for t in batch)
        if args.use_syntax:
            input_ids, input_lens, input_mask, label_ids, syntax_ids = batch
        else:
            input_ids, input_lens, input_mask, label_ids = batch

        if args.use_syntax:
            logits = model(input_ids,input_lens,input_mask,syntax_ids)
        else:
            logits = model(input_ids,input_lens,input_mask)
        
        predict_batch = torch.max(logits, 1)[1].data
        n_test_correct += torch.sum((predict_batch == label_ids.data))
        n_test_data += logits.size(0)
        for predict,label in zip(predict_batch,label_ids.data):
            results.append((label, predict))
    return (n_test_correct, n_test_data, n_test_correct/(1.0*n_test_data)),results


def main():
    args = get_args()
    # 句法标签
    syn_field = SyntaxField()
    args.len_syntax_dict = syn_field.len_syntax_dict
    
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    no_gpu = args.no_gpu
    n_gpu = 1
    device = torch.device("cuda", no_gpu)
    print("使用GPU%d" % no_gpu)
    print("                 ")
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

    processor = MrpcProcessor()
    output_mode = "classification"

    label_list = processor.get_labels()
    label_map = {label : i for i, label in enumerate(label_list)}
    # label数量
    num_labels = len(label_list)
    
    # 分字器
    tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)    
    bert_field = BertField('BERT',tokenizer=tokenizer)
    
    print("loaded tokenizer")
    
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs 
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    if args.maxpooling or args.avepooling:
        if args.syntax_hidden_embed:
            model = Bert_Classifier_Pooling_hidden(args)
        else:
            model = Bert_Classifier_Pooling(args)
    else:
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

    # dataloader定义
    dataloader = DataLoader(args.batch_size)
    ### eval dataloader
    if args.do_eval:
        best_precision = 0
        patience = 10
        iters_left = patience
        eval_examples = processor.get_dev_examples(args.data_dir)
        all_input_ids, all_input_lens, all_input_mask = bert_field.numericalize([example.text_a for example in eval_examples])
        
        if args.use_syntax:
            all_syntax_ids = syn_field.numericalize([example.text_syntax for example in eval_examples])
            all_syntax_embed = torch.load(args.syntax_embed_path%'valid') 

        all_label_ids = torch.tensor([label_map[f.label] for f in eval_examples], dtype=torch.long)

        # 输入控制
        if args.use_syntax:
            if args.syntax_hidden_embed:
                eval_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_embed, hidden_embed=args.syntax_hidden_embed)
            else:
                eval_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_ids, hidden_embed=args.syntax_hidden_embed)
        else:
            eval_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids)
        eval_batch_num = len(eval_batches)

    ### train dataloader
    if args.do_train:
        n_batch_correct = 0
        len_train_data = 0
        i_train_step = 0

        all_input_ids, all_input_lens, all_input_mask = bert_field.numericalize([example.text_a for example in train_examples])
        if args.use_syntax:
            all_syntax_ids = syn_field.numericalize([example.text_syntax for example in train_examples]) 
            all_syntax_embed = torch.load(args.syntax_embed_path%'train') 
        all_label_ids = torch.tensor([label_map[f.label] for f in train_examples], dtype=torch.long)
        # syntax information
        if args.use_syntax:
            if args.syntax_hidden_embed:
                train_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_embed,shuffle=True, hidden_embed=args.syntax_hidden_embed)
            else:
                train_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids, syntaxs=all_syntax_ids,shuffle=True, hidden_embed=args.syntax_hidden_embed)
        else:
            train_batches = dataloader.get_batches(all_input_ids,all_input_lens,all_input_mask, all_label_ids,shuffle=True)

        train_batch_num = len(train_batches)
        loss_fct = CrossEntropyLoss()

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            len_train_data, n_batch_correct = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_batches):
                i_train_step += 1
                
                batch = tuple(t.to(device) for t in batch)
                if args.use_syntax:
                    input_ids, input_lens, input_mask, label_ids, syntax_ids = batch
                else:
                    input_ids, input_lens, input_mask, label_ids = batch
                if args.use_syntax:
                    logits = model(input_ids,input_lens,input_mask,syntax_ids)
                else:
                    logits = model(input_ids,input_lens,input_mask)

                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                if epoch == 100:
                    print(torch.argmax(logits, dim=1))
                    print(label_ids)
                    from pdb import set_trace
                    set_trace()
                loss.backward()
                n_batch_correct += torch.sum((torch.max(logits, 1)[1].data == label_ids.data))
                len_train_data += logits.size(0)               
                if i_train_step % train_batch_num == 0:
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
            
            print('loss',tr_loss)
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
                for batch in eval_batches:
                    i_dev_times += 1
                    batch = tuple(t.to(device) for t in batch)
                    if args.use_syntax:
                        input_ids, input_lens, input_mask, label_ids, syntax_ids = batch
                    else:
                        input_ids, input_lens, input_mask, label_ids = batch

                    with torch.no_grad():
                        if args.use_syntax:
                            logits = model(input_ids,input_lens,input_mask,syntax_ids)
                        else:
                            logits = model(input_ids,input_lens,input_mask)
                    
                    if epoch == 100:
                        print(torch.argmax(logits, dim=1))
                        print(label_ids)
                        from pdb import set_trace
                        set_trace()
                    # create eval loss and other metric required by the task
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    
                    n_dev_batch_correct += torch.sum((torch.max(logits, 1)[1].data == label_ids.data))
                    len_dev_data += logits.size(0)  

                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)
                
                # from pdb import set_trace
                # set_trace()
                print('loss',eval_loss)
                logger.info('loss',eval_loss)
                P_dev = 1. * int(n_dev_batch_correct)/len_dev_data
                print()
                print("                                             ")
                print("-------------------------------------------------------------------")
                print("epoch:%d\tdev_Accuracy-----------------------%d/%d=%f\n"%(epoch,n_dev_batch_correct.data,len_dev_data,P_dev))
                logger.info("epoch:%d\tdev_Accuracy-----------------------%d/%d=%f\n"%(epoch,n_dev_batch_correct.data,len_dev_data,P_dev))
                if P_dev > best_precision:
                    best_precision = P_dev
                    iters_left = patience
                    if args.do_eval:
                        print("epoch %d saved\n"%epoch)
                        logger.info("epoch %d saved\n"%epoch)
                        torch.save(model.state_dict(),args.output_dir+'/model_best.pkl')
                else:
                    iters_left-=1
                    if iters_left == 0:
                        break       
                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                if output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                elif output_mode == "regression":
                    preds = np.squeeze(preds)
        print('lr', args.learning_rate, 'best_epoch', epoch-patience, 'best precision', best_precision)

        # test
        examples = processor.get_test_examples(args.data_dir)
        target, results = test(model, examples, bert_field, syn_field, dataloader, args, label_map, device)
        print("test_Accuracy-----------------------%d/%d=%f\n"%target)
        logger.info("test_Accuracy-----------------------%d/%d=%f\n"%target)
        with open(os.path.join(args.output_dir,'test_result.txt'),'w') as f:
            questions = [example.text_a for example in examples]
            for q,result in zip(questions,results):
                f.write("%s\t%d\t%d\n"%(q,result[0],result[1]))

if __name__ == "__main__":
    main()


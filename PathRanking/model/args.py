import os
from argparse import ArgumentParser

def get_args(mode='train'):
    parser = ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_path",
                        default='../../data/',
                        type=str)
    parser.add_argument("--bert_model", 
                        default='../../data/bert-base-chinese.tar.gz',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_vocab", 
                        default='../../data/bert-base-chinese-vocab.txt', 
                        type=str,
                        help="Bert Vocabulary")
    parser.add_argument("--output_dir",
                        default='saved',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_file",
                        default='../data/train.json',
                        type=str,
                        help="The train data. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--valid_file",
                        default='../data/valid.json',
                        type=str,
                        help="The valid data. Should contain the .json files (or other data files) for the task.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--neg_size",
                        default=5,
                        type=int,
                        help="Size of negative sample.")
    parser.add_argument("--neg_fix",
                        default=False,
                        action='store_true',
                        help="Whether not to fix neg sample.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--margin",
                        default=0.1,
                        type=float,
                        help="Margin for margin ranking loss.")
    parser.add_argument("--num_train_epochs",
                        default=100,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10,
                        type=int,
                        help="Stop training when nums of epochs not improving.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu",
                        type=str,
                        default='3',
                        help="use which gpu")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--optimizer",
                        type=str,
                        default='Adam',
                        help="choose optimizer")
    parser.add_argument("--model",
                        default='bert_comparing',
                        type=str,
                        choices=['bert_comparing','bert_sharecomparing']) 
    # model params
    parser.add_argument("--requires_grad",
                        action='store_true',
                        help="Whether not to fine tune Bert.")
    parser.add_argument("--maxpooling",
                        action='store_true',
                        help="Whether not to use maxpooling")
    parser.add_argument("--avepooling",
                        action='store_true',
                        help="Whether not to use avepooling")
    parser.add_argument("--bert_embedding_size",
                        default=768,
                        type=int)
    parser.add_argument("--hidden_dim",
                        default=300,
                        type=int)
    parser.add_argument("--syntax_dim",
                        default=800,
                        type=int,
                        help="dim for hidden syntax embedding")
    parser.add_argument("--lstm_layer",
                        default=1,
                        type=int)
    parser.add_argument("--bilstm",
                        action='store_false',
                        help='whether to use bilstm')
    parser.add_argument("--len_syntax_dict",
                        default=30,
                        type=int,
                        help="Num of syntax labels.")
    parser.add_argument("--dropout",
                        default=0.5,
                        type=float,
                        help='dropout rate for drop out layer.')
    if mode == 'predict':
         parser.add_argument("--model_path",
                        default='saved2/pytorch_model.bin',
                        type=str,
                        help="the path of trained model!")
         parser.add_argument("--input_file",
                        default='../cls_all_path/BERT_LSTM_maxpooling_embed/one_hop_cand_paths_ws_ent.json',
                        type=str,
                        help="the path of predict file!")
         parser.add_argument("--output_file",
                        default='',
                        type=str,
                        help="the path of predict file!")
         parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="batch size for test.")
         parser.add_argument("--topk",
                        default=1,
                        type=int,
                        help="topk paths while inferring.")
    args = parser.parse_args()
    return args

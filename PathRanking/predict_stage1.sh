DATADIR='../PreScreen/data/'
GPU=0
TOPK=5

MODEL='bert_sharecomparing'
MODELDIR='saved_sharebert/'
DATADIR2='merge/'

# predict
nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'one_hop_paths.json' --output_file $DATADIR$DATADIR2'one_hop_predict_path.json' --topk $TOPK >$DATADIR$DATADIR2'log_'$TOPK'.txt' &
echo 'Finish prescreen'
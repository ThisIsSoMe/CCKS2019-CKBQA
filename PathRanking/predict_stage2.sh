DATADIR='../PreScreen/data/'
GPU=0
TOPK=5

MODEL='bert_sharecomparing'
MODELDIR='saved_sharebert/'
DATADIR2='merge/'

nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR$DATADIR2'paths_all_merge.json' --output_file $DATADIR$DATADIR2'mix_predict_path.json' --topk 1 >'log_merge'$TOPK'.txt' &
echo 'Finish predcit'
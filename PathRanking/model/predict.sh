# bert_comparing模型
# DATADIR='saved_sharebert_negfix/'
DATADIR='../../PreScreen/data/'

GPU=0
TOPK=15
# DATADIR2='BERT_'$TOPK'_share_negfix/'
# DATADIR2='BERT_15_sharebert/'
# MODELDIR='saved_sharebert_negfix/'
# # MODELDIR='saved_adapt/'
# # MODEL='bert_sharecomparing'
# MODEL='bert_sharecomparing'
# # MODELDIR='saved/'
# # DATADIR2='BERT_15_share_negfix/'
A='../../PreScreen/data/'

MODEL='bert_sharecomparing'
MODELDIR='saved_sharebert/'
DATADIR2='merge/'

# predict
# nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'one_hop_paths.json' --output_file $DATADIR$DATADIR2'one_hop_predict_path.json' --topk $TOPK >$DATADIR$DATADIR2'log_'$TOPK'.txt' &
# echo 'Finish prescreen'

# nohup python -u ../../PreScreen/data/mix_paths.py --fn_in $A$DATADIR2"one_hop_predict_path.json" --fn_out $A$DATADIR2"mix_paths.json" >'log'$TOPK'.txt' &
# nohup python -u ../../PreScreen/data/merge_path.py --fn_in $A$DATADIR2"mix_paths.json" --fn_multi "multi_paths.json" --fn_out $DATADIR2"mix_paths_all.json" >'log'$TOPK'.txt' &
# echo 'Finish search path'

# nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR$DATADIR2'paths_all_merge.json' --output_file $DATADIR$DATADIR2'mix_predict_path.json' --topk 1 >'log_merge'$TOPK'.txt' &
# echo 'Finish predcit'


DATADIR='saved_char_feature/'
GPU=3
MODEL='bert_sharecomparing'
MODELDIR='saved/'
nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'one_hop_cand_paths_ws_ent.json' &
nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'multi_constraint_cand_paths_ws_ent.json' &
nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'two_hop_cand_paths_ws_ent.json' &
nohup python -u predict.py --gpu $GPU --learning_rate 1e-5 --margin 0.1 --model $MODEL --model_path $MODELDIR'pytorch_model.bin' --input_file $DATADIR'left_cand_paths_ws_ent.json' && MODELDIR='saved_char_feature/' && echo 'search answer...' && nohup python -u ../utils/search_ans.py --data_dir '../../PathRanking/model/'$MODELDIR &


# MODELDIR='saved_char_feature/' && python ../utils/search_ans.py --data_dir '../../PathRanking/model/'$MODELDIR

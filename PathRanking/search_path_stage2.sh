DATADIR='../PreScreen/data/'
TOPK=5
DATADIR2='merge/'

nohup python -u ../../PreScreen/data/mix_paths.py --fn_in $A$DATADIR2"one_hop_predict_path.json" --fn_out $A$DATADIR2"mix_paths.json" >'log'$TOPK'.txt' &
nohup python -u ../../PreScreen/data/merge_path.py --fn_in $A$DATADIR2"mix_paths.json" --fn_multi "multi_paths.json" --fn_out $DATADIR2"mix_paths_all.json" >'log'$TOPK'.txt' &
echo 'Finish search path'
# DATADIR="BERT_10_share_negfix/"
# DATADIR="BERT_15_negsample/"
# DATADIR="BERT_15_sharebert/"
# DATADIR="BERT_15_share_negfix/"
# prepare paths
# rm nohup.out
# nohup python -u mix_paths.py --fn_in $DATADIR"one_hop_predict_path.json" --fn_out $DATADIR"mix_paths.json" &
# python -u merge_path.py --fn_in $DATADIR"mix_paths.json" --fn_multi "multi_paths.json" --fn_out $DATADIR"mix_paths_all.json"

# after predict
DATADIR="merge/"
python search_ans.py --fn_in $DATADIR"mix_predict_path.json" --fn_out $DATADIR"mix_answer.json"

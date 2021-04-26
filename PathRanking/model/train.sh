# nohup python -u  main.py --requires_grad >saved/log.txt &

# nohup python -u main.py --gpu 3 --batch_size 8 --train_file '../data/all_train_data.json' --output_dir 'saved_rich_B8' --requires_grad >saved_rich_B8/log.txt &
# nohup python -u main.py --gpu 7 --batch_size 16 --train_file '../../PreScreen/preprocess/train.json' --valid_file '../../PreScreen/preprocess/valid_full_v2.json' --output_dir 'saved_mix' --requires_grad >saved_mix/log.txt &
# nohup python -u main.py --gpu 3 --model 'bert_sharecomparing' --batch_size 16 --train_file '../data/train.json' --valid_file '../data/valid.json' --output_dir 'saved_sharebert' --requires_grad >saved_sharebert/log.txt &
nohup python -u main.py --gpu 4 --model 'bert_sharecomparing' --neg_fix --batch_size 16 --train_file '../data/train.json' --valid_file '../data/valid.json' --output_dir 'saved_sharebert_negfix' --requires_grad >saved_sharebert_negfix/log.txt &2>1 &

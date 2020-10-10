export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

nohup python -u bert_main.py --config ccks_bert.cfg >ccks_log/log_1 2>&1 &
tail -f ccks_log/log_1


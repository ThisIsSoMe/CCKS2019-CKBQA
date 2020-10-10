from ans_tools import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search answer')
    parser.add_argument("--data_dir", default='', type=str)
    parser.add_argument("--fn_in", default='', type=str)
    parser.add_argument("--fn_out", default='', type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    # 原始数据：问题答案所有信息等
    fn_out = args.fn_out
    mask = '<E>'
    fn_in = args.fn_in
    with open(fn_in, 'r')as f:
        mix_paths = json.load(f)
    all_answer = {}
    for k,v in mix_paths.items():
        #v = v[0]
        if len(v) == 3:
            answer = one_hop_path2ans(v)
        elif len(v) == 4 and mask in v:
            answer = two_hop_path2ans(v)
        elif mask not in v:
            answer = multi_path2ans_one(v)
        else:
            answer = ''
        all_answer[k] = answer
        # from pdb import set_trace
        # set_trace()
    with open(fn_out, 'w')as f:
        json.dump(all_answer, f, ensure_ascii=False)
        

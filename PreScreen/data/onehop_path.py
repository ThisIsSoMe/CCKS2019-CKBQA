import sys
import json
import argparse
from argparse import ArgumentParser
sys.path.append('../')
from utils import *
read_con = pymysql.connect(host="192.168.126.143",port = 3337,user='root', password='pjzhang', database='ccks_2019',charset='utf8')
cur = read_con.cursor()

def get_paths_ent(mentions, ents):
    paths = []
    mask = '<E>'
    for m in mentions:
        this_m = '"'+m+'"'
        this_e = '<'+m+'>'

        try:
            record = from_value(this_m, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(mask, r, this_m) for r in rels])  

        try:
            record = from_value(this_e, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(mask, r, this_e) for r in rels])   

        try:
            record = from_entry(this_e, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(this_e, r, mask) for r in rels])   

    for e in ents:
        this_e = '<'+e+'>'
        this_m = '"%s"' % e
        try:
            record = from_entry(this_e, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(this_e, r, mask) for r in rels])

        try:
            record = from_value(this_e, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(mask, r, this_e) for r in rels])

        try:
            record = from_value(this_m, cur)
        except:
            record = []
        if len(record) > 0:
            rels = list(set([i[2] for i in record]))
            paths.extend([(mask, r, this_m) for r in rels]) 
    return paths

if __name__ == '__main__':
    # 输入文件
    parser = ArgumentParser(description='All One Hop For KBQA')
    parser.add_argument("--fn1", default='../data/test_format.json', type=str)
    parser.add_argument("--fn2", default="../NER/data/test_el_baike_top10.json", type=str)
    parser.add_argument("--fn_out", default="one_hop_paths.json", type=str)
    args = parser.parse_args()
    
    fn1 = args.fn1
    fn2 = args.fn2
    fn_out = args.fn_out  # 输出文件

    label_one = 1
    question_paths = []
    with open(fn1, 'r')as f1, open(fn2, 'r')as f2:
        test_all_data = json.load(f1)
        test_el_data = []
        for line in f2:
            line = line.strip()
            if line:
                piece = json.loads(line)
                test_el_data.append(piece)
        for one_data_1, one_data_2 in zip(test_all_data['questions'], test_el_data):
            one_question_paths = {}
            one_q_1 = one_data_1
            one_q_2, mentions, entities = one_data_2['question'], one_data_2['mentions'], one_data_2['ents']
            assert one_q_2 == one_q_1
            print(one_q_1)
            ents = []
            for k, v in entities.items():
                ents.extend(v)
            cand_paths = get_paths_ent(mentions, ents)
            one_question_paths['q'] = one_q_1
            one_question_paths['paths'] = list(set(cand_paths))
            # from pdb import set_trace
            # set_trace()
            question_paths.append(one_question_paths)
    json.dump(question_paths, open(fn_out, 'w'), ensure_ascii=False)
    print("问句数量：", len(question_paths))
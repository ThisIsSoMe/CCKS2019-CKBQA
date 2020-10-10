import json
import os
# pip install requests first
import requests

def segment(seqs):
    '''
    利用接口分词
    '''
    if not seqs:
        return []
    url = "http://192.168.126.171:5001/api"
    headers = {"Content-Type": "application/json; charset=UTF-8"}
    input_json = {"input_string":seqs, "ws": True, "pos": False, "dep": False}
    response = requests.post(url, data=json.dumps(input_json), headers=headers)
    outs = response.json()
    if 'words' in outs.keys():
        return outs['words']
    else:
        return [] 

if __name__ == '__main__':
    dir = '../cls_all_path/BERT_LSTM_maxpooling_embed'
    fns = ['one_hop_cand_paths_ent.json', 'two_hop_cand_paths_ent.json', 'multi_constraint_cand_paths_ent.json', 'left_cand_paths_ent.json']
    for fn in fns:
        path = os.path.join(dir, fn)
        output_path = os.path.join(dir, fn.replace('paths','paths_ws'))
        with open(path, 'r') as f, open(output_path, 'w')as fout:
            data_ws = []
            data_input = json.load(f)
            for line in data_input:
                q = line['q']
                seqs = segment([q.replace(' ','')])
                q_ws = seqs[0]
                paths = line['paths']
                seqs = [''.join(item).replace(' ','') for item in paths]
                paths_ws = segment(seqs)
                assert len(paths) == len(paths_ws)
                new_line = {'q':q, 'q_ws':q_ws, 'paths':paths, 'paths_ws':paths_ws}
                data_ws.append(new_line)
            json.dump(data_ws, fout, ensure_ascii=False)
            print('File', fn, 'finish')



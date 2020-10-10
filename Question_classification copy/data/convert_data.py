import json
# pip install requests first
import requests
#from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch

if __name__ == '__main__':
    # url = "http://192.168.126.171:5001/api"
    # headers = {"Content-Type": "application/json; charset=UTF-8"}
    # modes = ['train', 'valid', 'test']

    # for mode in modes:
    # # with open("../../data/%s.json"%mode,'r')as f, open('../../data/%s_format.json'%mode,'w')as f_format:
    # #     data = json.load(f)
    # #     [qindex_list,q_list,s_list,a_list,e_list,rel_list,mention_list,type_list] = data
    # #     new_data = {'qindex':qindex_list,'questions':q_list,'sqls':s_list,'answers':a_list,'ents':e_list,'triples':rel_list,'mentions':mention_list,'types':type_list}
    # #     json.dump(new_data, f_format, ensure_ascii=False)

    #     with open('../../data/%s_format.json'%mode,'r')as f, open('%s.tsv'%mode,'w')as fout:
    #         data = json.load(f)
    #         questions, types = data['questions'], data['types']
    #         for q,t in zip(questions, types):
    #             t = int(t)
    #             if t in [1,2]:
    #                 label = 1
    #             elif t in [4,5,7]:
    #                 label = 2
    #             elif t in [6,8,9,10]:
    #                 label = 3
    #             else:
    #                 label = 0
    #             fout.write('%s\t%s\n'%(q,label))

    # for mode in modes:
    #     seqs = []
    #     labels = []
    #     fn = '%s.tsv' % mode
    #     fn_out = '%s_syntax_word.tsv' % mode
    #     with open(fn, 'r', encoding='utf-8')as f:
    #         for line in f:
    #             one_words, label = line.strip().split('\t')[0].replace(' ',''), line.strip().split('\t')[1]
    #             seqs.append(one_words)
    #             labels.append(label)
    #     #input_json = {"words": words, "ws": True, "pos": True, "dep": True}
    #     input_json = {"input_string": seqs, "ws": True, "pos": True, "dep": True}
    #     response = requests.post(url, data=json.dumps(input_json), headers=headers)
    #     outs = response.json()
    #     print(outs.keys())
    #     with open(fn_out, 'w', encoding='utf-8')as f_out:
    #         for seq, syns, label in zip(outs['words'], outs['rels'], labels):
    #             f_out.write("%s\t%s\t%s\n" % (' '.join(seq), ' '.join(syns), label))

    modes = ['train', 'valid', 'test']
    for mode in modes:
        data = torch.load('%s.char.embed'%mode)
        print(len(data))

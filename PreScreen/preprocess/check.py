# import json

# with open("valid_full_v2.json",'r')as f:
#     data = json.load(f)
# questions = data['questions']
# gold = data['golds']
# negs = data['negs']
# # # query

# index = 0
# q = questions[index]
# g = gold[index]
# n = negs[index]
# print(q)
# print(g)
# print(len(n))
# for one in n:
#     if g[:-3] in one:
#         print(one)

##***************************************************************************
# modify the valid data
# cnt = 0
# new_negs = []
# for g, n in zip(gold, negs):
#     if g in n:
#         n.remove(g)
#         cnt = cnt + 1
#         print(cnt)
#     new_negs.append(n)
# cnt = 0
# for g, n in zip(gold, new_negs):
#     if g in n:
#         cnt += 1
# print(cnt)
# assert len(negs) == len(new_negs)

# new_data ={}
# new_data['questions'] = questions
# new_data['golds'] = gold
# new_data['negs'] = new_negs
# with open("valid_full_v2.json",'w')as f:
#     json.dump(new_data, f, ensure_ascii = False)
import torch
import jieba
import json

with open('../data/one_hop_paths.json','r')as f1:
    one_hop_data = json.load(f1)
    embed_dict = {}
    embed_data = torch.load('test.char.embed')
    for line, embedding in zip(one_hop_data, embed_data):
        q = line['q']
        q_ws = [i for i in jieba.cut(q)]
        assert len(q_ws) == embedding.shape[0]
        embed_dict[q] = embedding
    torch.save(embed_dict, 'test.char.embed.dict')

import json
folds=['BERT_2','BERT_5','BERT_share','BERT_15','BERT_20','BERT_TandA']
for fold in folds:
    with open(fold+"/mix_paths_all.json",'r')as f:
        data = json.load(f)
        all = 0
        num = 0
        maxn = 0
        for line in data:
            v = line['paths']
            v = [tuple(i) for i in v]

            # v = list(set(v))
            all += len(v)
            if len(v) > 0:
                num += 1
            if len(v) > maxn:
                maxn = len(v)
    print(fold)
    print('average number of path:')
    print(all/766)
    print('num',num)
    print('max',maxn)
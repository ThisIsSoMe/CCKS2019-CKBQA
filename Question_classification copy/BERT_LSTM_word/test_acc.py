multi_label_dict = {}
with open('../../data/multi_label_result.txt','r')as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        else:
            q = line.split('\t')[0][:-1]
            multi_label_dict[q.replace(' ','')] = 1

print(multi_label_dict)
#
fn = 'saved_syntax_word_embed_2/test_result.txt'
holder = ['[', ']', "'", ',', ' ']
with open(fn, 'r')as f:
    total = 0
    right = 0
    for line in f:
        line= line.strip()
        if not line:
            continue
        items = line.split('\t')
        q = items[0]
        for i in holder:
            q = q.replace(i,'')
        #print(q)
        gold, pred = int(items[-2]),int(items[-1])
        gold = [gold]
        if q in multi_label_dict:
            gold.append(multi_label_dict[q])
        total += 1
        if pred in gold:
            right += 1
    print('acc %d/%d=%f'%(right, total, 1.0*right/total))

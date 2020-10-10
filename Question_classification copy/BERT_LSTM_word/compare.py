fn1 = 'saved_word/test_result.txt'
fn2 = 'saved_syntax_word_embed_2/test_result.txt'
#label = '0'

for label in ['1', '2', '3','0']:
    num1 = 0
    num2 = 0
    with open(fn1,'r')as f1, open(fn2, 'r')as f2:
        for line1, line2 in zip(f1,f2):
            line1 = line1.strip()
            line2 = line2.strip()
            if line1 and line2:
                q = line1.split('\t')[0]
                for symbol in ['[',']',"'",',']:
                    q = q.replace(symbol,'')
                gold1 = line1.split('\t')[-2]
                gold2 = line2.split('\t')[-2]
                pred1 = line1.split('\t')[-1]
                pred2 = line2.split('\t')[-1]
                assert gold1 == gold2

                if gold1 == label and pred1 != pred2:
                    if pred2 == label:
                        print('file2:%s'%''.join(q))
                        num2 += 1
                    elif pred1 == label:
                        print('file1:%s'%''.join(q))
                        num1 += 1
    print('gold label:%s, file1 right %d, file2 right %d'%(label, num1, num2))

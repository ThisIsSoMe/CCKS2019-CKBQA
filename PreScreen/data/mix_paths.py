import sys
import json
import argparse
from argparse import ArgumentParser
sys.path.append('../../')
from utils import *
import pymysql
read_con = pymysql.connect(host="192.168.126.143",port = 3337,user='root', password='pjzhang', database='ccks_2019',charset='utf8')
cur = read_con.cursor()

def select_twohop_path(seq, onehop_path, cur, mask='<E>'):
    cand_paths = []
    for path in onehop_path:
        if mask == path[0]:  # 反向查询
            focus, rel = path[-1], path[1]
            record = search_ans(focus, rel, cur, reverse=True)
            ents = [line[1] for line in record]
            for e in ents:
                try:
                    record = from_entry(e,cur)
                except:
                    continue

                path2 = list(set([(i[2], i[3]) for i in record]))
                
                cand_paths.extend([(focus, rel, p2[0], mask) for p2 in path2 if p2[1] != focus and rel != p2[0]]) # 不找回自己
                
                cand_paths.extend([(focus,rel,p2[1],p2[0]) for p2 in path2 if p2[1] != focus and rel != p2[0] and del_des(p2[1])[1:-1] in seq]) # 答案结点在句子内出现

                
                try:
                    record = from_value(e, cur)
                except:
                    continue
                path2 = list(set([(i[2], i[1]) for i in record]))
                # 该情况下不反向查找
                # cand_paths.extend([(focus, rel, p2[0], mask) for p2 in path2 if p2[1] != focus and rel != p2[0]]) # 不找回自己
                cand_paths.extend([(focus, rel, p2[1], p2[0]) for p2 in path2 if p2[1] != focus and rel != p2[0] and del_des(p2[1])[1:-1] in seq]) # 答案结点在句子内出现

        elif mask == path[2]: # 正向查询
            focus, rel = path[0], path[1]
            record = search_ans(path[0], path[1], cur, reverse=False)
            ents = [line[3] for line in record]

            for e in ents:
                if e[0] != '"':
                    try:
                        record = from_entry(e,cur)
                    except:
                        continue

                    path2 = list(set([(i[2], i[3]) for i in record]))
                    
                    cand_paths.extend([(focus, rel, p2[0], mask) for p2 in path2 if p2[1] != focus and rel != p2[0]]) # 不找回自己
                    
                    cand_paths.extend([(focus, rel, p2[1], p2[0]) for p2 in path2 if p2[1] != focus and rel != p2[0] and del_des(p2[1])[1:-1] in seq]) # 答案结点在句子内出现

                try:
                    record = from_value(e, cur)
                except:
                    continue

                path2 = list(set([(i[2], i[1]) for i in record]))

                cand_paths.extend([(focus, rel, p2[0], mask) for p2 in path2 if p2[1] != focus and rel != p2[0]]) # 不找回自己
                
                cand_paths.extend([(focus, rel, p2[1], p2[0]) for p2 in path2 if p2[1] != focus and rel != p2[0] and del_des(p2[1])[1:-1] in seq]) # 答案结点在句子内出现
    return list(set(cand_paths))

def intersect_items(str1,str2,cur):
    # str1,str带引号和尖括号标志
    items1,items2=[],[]
    items1.extend([i[1] for i in from_value(str1,cur)])
    items2.extend([i[1] for i in from_value(str2,cur)])
    if str1[0]=='<':
        items1.extend([i[3] for i in from_entry(str1,cur)])
    if str2[0]=='<':
        items2.extend([i[3] for i in from_entry(str2,cur)])
    # 取交集
    intersection=[]
    for e in items1:
        if e in items2:
            intersection.append(e)
    return intersection

def link_items(str1,cur):
    items1=[]
    items1.extend([i[1] for i in from_value(str1,cur)])
    if str1[0]=='<':
        items1.extend([normalize(i[3]) for i in from_entry(str1,cur)])
    return items1

#mention聚类
def clustering(question,mentions,gender=False):
    clusters=[]
    mentions.sort(key=lambda x:-len(x))
    for mention in mentions:
        # 一层判断，应对同义词替换后找不到mention位置的情况
        if mention not in question:
            continue
        flag=False # 标记有没有找到相应的分类
        start=question.index(mention)
        end=start+len(mention)-1
        # 每一个分类
        for i,c in enumerate(clusters):
            for m in c: 
                start_m=question.index(m)
                end_m=start_m+len(m)-1
                if (start>=start_m and start<=end_m) or (end>=start_m and end<=end_m) or (start<start_m and end>end_m):
                    #set_trace()
                    clusters[i].append(mention)
                    flag=True
                    break
            if flag:
                break
        if not flag:
            clusters.append([mention])
    new_clusters=[]
    for clu in clusters:
        if len(clu[0])==1:
            if gender:
                if clu[0] in ['男','女']:
                    new_clusters.append(clu)
            else:
                continue
        else:
            new_clusters.append(clu)
    # mention_longest=[cluster[0] for cluster in new_clusters]

    # this_question=question
    # for word in mention_longest:
    #     this_question=this_question.replace(word,' '+word+'/ ')
    return new_clusters

def clustering_with_syntax(question,mentions,syntax_out,gender=False):
    clusters=[]
    
    my_syntax_label='att'
    pattern="%d_%d_att"
    
    mentions.sort(key=lambda x:-len(x))
    ws=syntax_out.split('\t')[0].split(' ')
    syntax=['_'.join(string.split('_')[:-1]) for string in syntax_out.split('\t')[-1].split(' ')]
    for mention in mentions:
        # 一层判断，应对同义词替换后找不到mention位置的情况
        if mention not in question:
            continue
        flag=False # 标记有没有找到相应的分类
        start=question.index(mention)
        end=start+len(mention)-1
        # 每一个分类
        for i,c in enumerate(clusters):
            for m in c: 
                start_m=question.index(m)
                end_m=start_m+len(m)-1
                if (start>=start_m and start<=end_m) or (end>=start_m and end<=end_m) or (start<start_m and end>end_m):
                    #set_trace()
                    clusters[i].append(mention)
                    flag=True
                    break
                else:
                    if m in ws and mention in ws:
                        i1=ws.index(m)
                        i2=ws.index(mention)
                        if pattern%(i1+1,i2+1) in syntax or pattern%(i2+1,i1+1) in syntax:
                            clusters[i].append(mention)
                            flag=True
                            break
            if flag:
                break
        if not flag:
            clusters.append([mention])
    new_clusters=[]
    for clu in clusters:
        if len(clu[0])==1:
            if gender:
                if clu[0] in ['男','女']:
                    new_clusters.append(clu)
            else:
                continue
        else:
            new_clusters.append(clu)
    return new_clusters
    
    
def intersect_2path(str1_list,str2_list,cur):
    paths=[]
    for str1 in str1_list:
        for str2 in str2_list:
            
            # 作为尾实体寻找
            items1,items2=[],[]
            items1.extend([(str1,i[2],i[1]) for i in from_value(str1,cur)])
            items2.extend([(str2,i[2],i[1]) for i in from_value(str2,cur)])
            
            # 作为头实体寻找
            if str1[0]=='<':
                items1.extend([(str1,i[2],i[3]) for i in from_entry(str1,cur)])
            if str2[0]=='<':
                items2.extend([(str2,i[2],i[3]) for i in from_entry(str2,cur)])
                
            for a in items1:
                item1=a[2]
                for b in items2:
                    item2=b[2]
                    if item1==item2:
                        paths.append((a[0],a[1],b[0],b[1]))
    return list(set(paths))

def intersect_3path(str1_list,str2_list,str3_list,cur):
    paths=[]
    for str1 in str1_list:
        for str2 in str2_list:
            for str3 in str3_list:
                # 作为尾实体寻找
                items1,items2,items3=[],[],[]
                items1.extend([(str1,i[2],i[1]) for i in from_value(str1,cur)])
                items2.extend([(str2,i[2],i[1]) for i in from_value(str2,cur)])
                items3.extend([(str3,i[2],i[1]) for i in from_value(str3,cur)])
            
                # 作为头实体寻找
                if str1[0]=='<':
                    items1.extend([(str1,i[2],i[3]) for i in from_entry(str1,cur)])
                if str2[0]=='<':
                    items2.extend([(str2,i[2],i[3]) for i in from_entry(str2,cur)])
                if str3[0]=='<':
                    items3.extend([(str3,i[2],i[3]) for i in from_entry(str3,cur)])
                
                for a in items1:
                    item1=a[2]
                    for b in items2:
                        item2=b[2]
                        if item1!=item2:
                            continue
                        for c in items3:
                            item3=c[2]
                            if item1==item3:
                                #set_trace()
                                paths.append((a[0],a[1],b[0],b[1],c[0],c[1]))
    return list(set(paths))


def get_paths_ent_multi(que, mentions, ents, cur, mask='<E>'):
    this_paths = []
    clusters=clustering(que,mentions)  # 聚类
    topk=100
    if len(clusters)==2:
        ents=[]
        print(que,mentions)
        for ment in clusters: # 对一个类别cluster
            this_ent=[]
            for m in ment: # 对单个cluster内的单个mention
                ent1=[(line[2],line[3]) for line in get_entry(m,cur)]
                ent1.sort(key=lambda x:x[1])
                this_ent.extend(['<%s>'%one[0] for one in ent1[:topk]])

                this_ent.append('"%s"'%m)
                this_ent.append('<%s>'%m)
            this_ent=sorted(this_ent,key=lambda x:len(x[0]))     
            ents.append(this_ent)
        #print(key,ents)
        # 取交集
        paths=intersect_2path(ents[0],ents[1],cur)
        this_paths=paths
    elif len(clusters)==3:
        ents=[] 
        for ment in clusters: # 对一个类别cluster
            this_ent=[]
            for m in ment: # 对单个cluster内的单个mention
                ent1=[(line[2],line[3]) for line in get_entry(m,cur)]
                ent1.sort(key=lambda x:x[1])
                this_ent.extend(['<%s>'%one[0] for one in ent1[:topk]])
                this_ent.append('"%s"'%m)
            this_ent=sorted(this_ent,key=lambda x:len(x[0]))     
            ents.append(this_ent)
        # 取交集
        paths=intersect_3path(ents[0],ents[1],ents[2],cur)
        this_paths=paths
    
    # if this_paths:
    #   return this_paths  
        
    # entities = mentions
    # for v in ents:
    #     entities.append(v)
    # entities = list(set(entities))
    # for one_e in entities:
    #     e = "<%s>" % one_e
    #     value = '"%s"' % one_e
    #     # 从实体开始搜索
    #     # 头找尾
    #     record = []
    #     try:
    #         record = list(from_entry(e,cur))
    #     except:
    #         continue
    #     path1 = list(set([(i[1],i[2],i[3]) for i in record if i[1][0] == '<' and i[3][0]=='<']))

    #     # 头尾尾
    #     for p1 in path1:
    #         tail = p1[2]
    #         record = []
    #         try:
    #             record = from_entry(tail,cur)
    #         except:
    #             continue
    #         path2 = list(set([(i[3],i[2]) for i in record]))
    #         for p2 in path2:
    #             if p2[0] != e and p1[0][1:-1] != p2[0][1:-1]:
    #                 if p2[0][1:-1] in que or del_des(p2[0])[1:-1] in que: 
    #                     this_paths.append((p1[0],p1[1],p2[0],p2[1]))
    #                 else:
    #                     pass
                        
    #     # 头尾头
    #     for p1 in path1:
    #         tail = p1[2]
    #         record = []
    #         try:
    #             record = from_value(tail, cur)
    #         except:
    #             continue
    #         path2 = list(set([(i[1], i[2]) for i in record]))                        
    #         for p2 in path2:
    #             if p2[0] != e and p1[0][1:-1] != p2[0][1:-1]:
    #                 if p2[0][1:-1] in que or del_des(p2[0])[1:-1] in que: 
    #                     this_paths.append((p1[0],p1[1],p2[0],p2[1]))
    #                 else:
    #                     pass
                    
    #     # 从尾部（属性值/实体）开始搜索
    #     record = []
    #     path1 = []
    #     try:
    #         record=list(from_value(e,cur))
    #         path1.extend(list(set([(i[3],i[2],i[1]) for i in record])))
    #     except:
    #         pass       
    #     try:
    #         record = list(from_value(value,cur))
    #         path1.extend(list(set([(i[3],i[2],i[1]) for i in record])))
    #     except:
    #         pass
    #     for p1 in path1:
    #         tail = p1[2]
    #         record = []
    #         try:
    #             record = from_entry(tail,cur)
    #         except:
    #             continue
    #         path2 = list(set([(i[3],i[2]) for i in record]))                        
    #         for p2 in path2:
    #             if p2[0] != e and p1[0][1:-1] != p2[0][1:-1]:
    #                 if p2[0][1:-1] in que or del_des(p2[0])[1:-1] in que: 
    #                     this_paths.append((p1[0],p1[1],p2[0],p2[1]))
    #                 else:
    #                     pass
            
    #         record = []
    #         try:
    #             record = from_value(tail,cur)
    #         except:
    #             continue
    #         path2 = list(set([(i[1],i[2]) for i in record]))                        
    #         for p2 in path2:
    #             if p2[0] != e and p1[0][1:-1] != p2[0][1:-1]:
    #                 if p2[0][1:-1] in que or del_des(p2[0])[1:-1] in que: 
    #                     this_paths.append((p1[0],p1[1],p2[0],p2[1]))
    #                 else:
    #                     pass
    return list(set(this_paths)) 


if __name__ == "__main__":
    parser = ArgumentParser(description='Next Hop For KBQA')
    parser.add_argument("--fn_test", default='../../data/test_format.json', type=str)
    parser.add_argument("--fn_el", default="../../NER/data/test_el_baike_top10.json", type=str)
    parser.add_argument("--fn_in", default="lstm_syntax/one_hop_predict_path.json", type=str)
    parser.add_argument("--fn_out", default="lstm_syntax/mix_paths.json", type=str)
    args = parser.parse_args()

    with open(args.fn_in, 'r')as f:
        onehop_paths = json.load(f)

    with open(args.fn_el, 'r')as f:
        EL_data = []
        for line in f:
            line = line.strip()
            if line:
                EL_data.append(json.loads(line))

    with open(args.fn_test, 'r')as f:
        test_data = json.load(f)
    
    with open("multi_paths.json",'r')as f:
        multi_data = json.load(f)

    seq_path_dict = onehop_paths
    # seq_path_dict = {}
    # for line in onehop_paths:
    #     seq = line['q']
    #     onehop_path = line['paths']
    #     seq_path_dict[seq] = onehop_path

    mix_paths = []

    multi_paths = {}
    for line_el, seq in zip(EL_data, test_data['questions']):
    # multi2one path only    
    #     mentions = line_el["mentions"]
    #     ents = []
    #     for k,v in line_el["ents"].items():
    #         ents.extend(v)
    #     multi_path = get_paths_ent_multi(seq, mentions, ents, cur)
    #     print("q:%s\tnum of paths:%d"%(seq, len(multi_path)))
    #     multi_paths[seq] = multi_path
    
    # with open(args.fn_out, 'w') as f:
    #     json.dump(multi_paths, f, ensure_ascii=False)

    # all paths included
        if seq in seq_path_dict.keys():
            onehop_path = seq_path_dict[seq]
            all_paths = onehop_path
            mentions = line_el["mentions"]
            ents = []
            for k,v in line_el["ents"].items():
                ents.extend(v)
            twohop_path = select_twohop_path(seq, onehop_path, cur)
            # multi_path = get_paths_ent_multi(seq, mentions, ents, cur)
            multi_path = multi_data[seq]
            all_paths.extend(twohop_path)
            all_paths.extend(multi_path)

            one_data = {}
            one_data['q'] = seq
            one_data['paths'] = all_paths
            mix_paths.append(one_data)

            print("q:%s\tnum of paths:%d"%(seq, len(all_paths)))
    
    with open(args.fn_out, 'w') as f:
        json.dump(mix_paths, f, ensure_ascii=False)
    
    


        
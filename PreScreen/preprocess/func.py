import json
import random
import sys
sys.path.append('../../')
from utils import *

def get_one_hop_path(cur, mentions, PAD='<E>', topk=None, addition=None, addition_bound=None,gold=None, des=False):
    '''
    mentions: 搜索起点
    gold: gold path
    topk: 移除gold path，截取k个路径
    addition:额外的关系列表
    addition_bouund:扩充至最小数量
    '''
    paths = []
    for m in mentions:
        this_m='"'+m+'"'
        this_e = '<%s>' % m
        this_e_n = del_des(this_e)
        try:
            record=from_value(this_e,cur)
        except:
            record=[]

        if len(record) > 0:
            rels=list(set([i[2] for i in record]))
            paths.extend([(PAD, r, this_e if des else this_e_n) for r in rels])  

        try:
            record=from_value(this_m,cur)
        except:
            record=[]

        if len(record)>0:
            rels=list(set([i[2] for i in record]))
            paths.extend([(PAD, r, this_m) for r in rels])   
    
        try:
            record=from_entry(this_e,cur)
        except:
            record=[]

        if len(record)>0:
            rels=list(set([i[2] for i in record]))
            paths.extend([(this_e if des else this_e_n, r, PAD) for r in rels])   
    
    paths=list(set(paths))
    if gold:
        if gold in paths:
            paths.remove(gold)
    if topk:
        random.shuffle(paths)
        paths = paths[:topk]
    if topk and addition and addition_bound and len(paths)<addition_bound:
            random.shuffle(addition)
            while len(paths) < addition_bound:
                i = random.randint(0,len(addition)-1)
                r1 = addition[i]
                paths.append((this_e if des else this_e_n, r1, PAD))
    return paths

def get_two_hop_path(cur, mentions, PAD='<E>', topk=None, addition=None, addition_bound=None, gold=None, des=False):
    '''
    mentions: 搜索起点
    gold: gold path
    topk: 移除gold path，截取k个路径
    addition:额外的关系列表
    addition_bouund:扩充至最小数量
    '''
    paths = []
    for m in mentions:
        this_m = '"'+m+'"'
        this_e = '<%s>' % m
        this_e_n = del_des(this_e)
        
        # *头尾
        reverse_record = []
        record = []
        copy_record = []
        try:               
            reverse_record = from_value(this_e, cur)
            if reverse_record:
                record += tuple([('',this_e if des else this_e_n,item[2],item[1]) for item in reverse_record])
        except:
            pass
        
        try:
            reverse_record = from_value(this_m, cur)
            if reverse_record:
                record += tuple([('',this_m,item[2],item[1]) for item in reverse_record])
        except:
            pass
        
        try:
            reverse_record = from_entry(this_e, cur)
            copy_record = reverse_record
            if reverse_record:
                record += tuple([('',this_e if des else this_e_n,item[2],item[3]) for item in reverse_record])
        except:
            pass
        
        for line in record:
            mid = line[3]
            head = line[1]
            rel1 = line[2]
            try:
                reverse_record = from_entry(mid, cur)
                rel2 = list(set([item[2] for item in reverse_record if item[3] not in [this_m,this_e]]))
                paths.extend([(head, rel1, r, PAD) for r in rel2[:topk]])
            except:
                pass
        
        paths = list(set(paths))
        if topk:
            random.shuffle(paths)
            paths = paths[:topk]
        
        # 头尾头
        record = []
        if copy_record:
            reocrd = tuple([('',this_e if des else this_e_n,item[2],item[3]) for item in copy_record])
 
        for line in record:
            mid = line[3]
            head = line[1]
            rel1 = line[2]
            rel2 = []
            if mid[0] == '<':
                try:
                    reverse_record = from_entry(mid, cur)
                    rel2.extend(list(set([item[2] for item in reverse_record if item[3] not in [this_m,this_e]])))
                except:
                    pass
            rel2 = list(set(rel2))
            paths.extend([(head, rel1, r, PAD) for r in rel2[:topk]])
        
        paths = list(set(paths))
        if gold:
            if gold in paths:
                paths.remove(gold)
        if topk:
            random.shuffle(paths)
            paths = paths[:topk]
        if topk and addition and addition_bound and len(paths)<addition_bound:
            random.shuffle(addition)
            while len(paths)<addition_bound:
                i = random.randint(0,len(addition)-1)
                j = random.randint(0,len(addition)-1)
                r1 = addition[i]
                r2 = addition[j]
                if i != j:
                    paths.append((this_e if des else this_e_n, r1, r2, PAD))
        return paths
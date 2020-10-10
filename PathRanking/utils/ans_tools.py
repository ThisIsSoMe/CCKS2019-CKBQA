# 搜索数据库找答案
import sys
import json
from pdb import set_trace

sys.path.append('../../')
from utils import *
read_con = pymysql.connect(host="192.168.126.143",port = 3337,user='root', password='pjzhang', database='ccks_2019',charset='utf8')
cur = read_con.cursor()

def multi_path2ans(fns):
    answer={}
    path_questions={}
    for fn in fns:
        out_merge=json.load(open(fn,'r'))
        fn_out=fn.replace('path','ans')
        for key,value in out_merge.items():
            out=[]
            path_questions[key]=value
            out1=[]
            out1.extend([line[3] for line in search_ans(value[0],value[1],cur)])
            out1.extend([line[1] for line in search_ans(value[0],value[1],cur,reverse=True)])
            out2=[]
            out2.extend([line[3] for line in search_ans(value[2],value[3],cur)])
            out2.extend([line[1] for line in search_ans(value[2],value[3],cur,reverse=True)])

            if len(value)==6:
                out3=[]
                out3.extend([line[3] for line in search_ans(value[4],value[5],cur)])
                out3.extend([line[1] for line in search_ans(value[4],value[5],cur,reverse=True)])
            for item in out1:
                if item in out2 and len(value)==6 and item in out3: 
                    out.append(item)
                elif item in out2 and len(value)==4: 
                    out.append(item)
            answer[key]=list(set(out))
        json.dump(answer,open(fn_out,'w'),ensure_ascii=False)
    return answer

def one_hop_path2ans(fn):
    
    fn_out=fn.replace('path','ans')
    data_input=json.load(open(fn,'r'))
    answer={}
    path_questions = {}
    x1 = 0
    x2 = 0
    x3 = 0
    PAD='<E>'
    question_multi_path={}
    for key,value in data_input.items():   
        path_questions[key]=value
        origin_tri=value
        if len(origin_tri)==0:
            print(key,value)
            continue

        if PAD==origin_tri[0]:
            x1+=1
            if origin_tri[2][0]=='"':
                e=origin_tri[2]
                rel=rel=origin_tri[1]
                ans=search_ans(e,rel,cur,reverse=True)
                answer[key]=[j[1] for j in ans]
#                     if len(ans)==1 and ans[0][1][1:-1] in key:
#                         answer[key]=[ans[0][3]]
                continue
#                 ent=get_entry(origin_tri[2][1:-1],cur)
#                 # 排序
#                 tmp1=origin_tri[2][1:-1]       
#                 # 加入mention对应的实体
#                 ents=[(tmp1,0)]
#                 ents.extend([(i[2],int(i[-1])) for i in ent])

#                 ents=list(set(ents))       
#                 ents=sorted(ents,key=lambda x:x[1])
            ents=[(origin_tri[2][1:-1],0)]
            rel=origin_tri[1]
            for e in ents:
                if e[0][0]!='<':
                    e='<'+e[0]+'>'
                else:
                    e=e[0]
                ans=search_ans(e,rel,cur,reverse=True)            
                if len(ans)==0:
                    pass
                else:
                    answer[key]=[j[1] for j in ans]
#                         if len(ans)==1 and ans[0][1][1:-1] in key:
#                             answer[key]=[ans[0][3]]
                    break     
            if key not in answer.keys():
                print(key,value,ents)

        elif PAD==origin_tri[2] and origin_tri[0]!='<None>':
            x2+=1
#                 ent=get_entry(origin_tri[0][1:-1],cur)       
#                 tmp1=origin_tri[0][1:-1]       
#                 # 加入mention对应的实体
#                 ents=[(tmp1,0)]
#                 ents.extend([(i[2],int(i[-1])) for i in ent])

#                 ents=list(set(ents))        
#                 ents=sorted(ents,key=lambda x:x[1])

            ents=[(origin_tri[0][1:-1],0)]

            rel=origin_tri[1]        

            #按知名度进行答案搜索
            for e in ents:
                if e[0][0]!='<':
                    e='<'+e[0]+'>'
                else:
                    e=e[0]
                ans=search_ans(e,rel,cur,reverse=False)

                if len(ans)==0:
                    pass
                else:
                    answer[key]=[j[3] for j in ans]
                    break
        else:
            print("ERROR question:",key,value)
        if not ans:
            print("ERROR question:",key,value)
    print("正向：",x1,"反向",x2)
    print('总共：',len(answer.keys()))
    fff = open(fn_out,"w")
    json.dump(answer,fff,ensure_ascii=False)
    return answer

def two_hop_path2ans(fns):
    for fn in fns:
        print(fn+':\n')
        data_input=json.load(open(fn,'r'))
        fn_out=fn.replace('path','ans')
        answer={}
        path_questions = {}
        mask='<E>'
        question_multi_path={}
        for key,value in data_input.items():   
            path_questions[key]=value
            items=value
            if len(items)==0:
                print(key,value)
                continue
                
            if len(items)==3:
                if items[0]==mask:
                    one_ans=search_ans(items[2],items[1],cur,reverse=True)
                    answer[key]=list(set([it[1] for it in one_ans]))
                else:
                    one_ans=search_ans(items[0],items[1],cur,reverse=False)
                    answer[key]=list(set([it[3] for it in one_ans]))
                continue
            
            if '<E>' not in items and len(items)==4:
                out=[]
                out1=[]
                out1.extend([line[3] for line in search_ans(value[0],value[1],cur)])
                out1.extend([line[1] for line in search_ans(value[0],value[1],cur,reverse=True)])
    #             if key=='"光武中兴"说的是哪位皇帝？':
    #                 set_trace()
                out2=[]
                out2.extend([line[3] for line in search_ans(value[2],value[3],cur)])
                out2.extend([line[1] for line in search_ans(value[2],value[3],cur,reverse=True)])

                for item in out1:
                    if item in out2: 
                        out.append(item)
                answer[key]=list(set(out))
                continue
            
            if items[0][0]=='<':
                mention=items[0][1:-1]
                ent=get_entry(mention,cur)
                # 排序       
                # 加入mention对应的实体
                ents=[(mention,0)]
                ents.extend([(i[2],int(i[-1])) for i in ent])
                ents=list(set(ents))       
                ents=sorted(ents,key=lambda x:x[1])
            else:
                ents=[items[0]]

            #ents=[(items[0],0)]
            final_ans=[]
            for e in ents:
                if e[0]=='"':
                    one_ans=search_ans(e,items[1],cur,reverse=True)
                    for ans in [k[1] for k in one_ans]:
                        two_ans=search_ans(ans,items[2],cur,reverse=False)
                        final_ans.extend([i[3] for i in two_ans])
                else:
                    e='<'+e[0]+'>'
                    one_ans=search_ans(e,items[1],cur,reverse=False)
                    one_out=[k[3] for k in one_ans]
                    one_ans=search_ans(e,items[1],cur,reverse=True)
                    one_out.extend([k[1] for k in one_ans])
                    for ans in one_out:
                        two_ans=search_ans(ans,items[2],cur,reverse=False)
                        if not two_ans:
                            two_ans=search_ans(ans,items[2],cur,reverse=True)
                        final_ans.extend([i[3] for i in two_ans])                      
                if not final_ans:
                    continue
                answer[key]=list(set(final_ans))
                break  
            if not final_ans:
                print(key,value,final_ans)
        fff = open(fn_out,"w")
        json.dump(answer,fff,ensure_ascii=False)
        return answer

# 单个问句的prf值
def one_value(pred,gold):
    pred=set(pred)
    gold=set(gold)
    inter=pred.intersection(gold)
    if len(inter)==0:
        p,r,f=0.0,0.0,0.0
    else:
        p=float(len(inter)/len(pred))
        r=len(inter)/len(gold)
        f=2*p*r/(p+r)
    return p,r,f
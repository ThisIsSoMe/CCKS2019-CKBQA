import pymysql
import pandas as pd
import gc
import jieba
import json
import re

read_con = pymysql.connect(host="192.168.126.143",port = 3337, user='root', password='pjzhang', database='ccks_2019',charset='utf8')
from sqlalchemy import create_engine
save_con = create_engine('mysql+pymysql://root:pjzhang@192.168.126.143:3337/ccks_2019?charset=utf8')
cur = read_con.cursor()

END = '\n'

# 根据三元组表pkubase查找实体，逆向最大匹配算法
def search_entity_pkubase_backward(line, max_string = True):
    end = len(line)
    entity = []
    new_line = ''
    while end > 0:
        begin = 0
#         print('end', end)
        while (end > begin):
#             print('begin', begin, end = ' ')
            word = line[begin: end]
            # print('w', word)
            sql = "select count(*) from `pkubase` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            begin += 1
            if (data[0][0] > 0):
#                 print(word)
                entity.append(word)
                #                 print('data', data[0:3])
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
            sql = "select count(*) from `pkubase` where `value`='%s' or `value`='<%s>' or `value`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                #                 print(word)
                entity.append(word)
                #                 print('data', data[0:3])
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
            sql = "select count(*) from `pkubase` where `prop`='%s' or `prop`='<%s>' or `prop`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if(data[0][0] > 0):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
            sql = "select count(*) from `pkuorder` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
        end -= 1
    entity.reverse()
#     print('entity')
    # entity = delete_stopwords(entity)
    return entity

# 根据三元组表pkubase查找实体,正向最大匹配算法
def search_entity_pkubase_forward(line, max_string = True):
    begin = 0
    entity = []
    while begin < len(line):
        end = len(line)
        # print('begin', begin)
        while(end > begin):
            word = line[begin: end]
            # print('end', end, end = ' ')
            sql = "select count(*) from `pkubase` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            end -= 1
            if(data[0][0] > 0):
                # print('word', word)
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    begin = end
                    break
            sql = "select count(*) from `pkubase` where `value`='%s' or `value`='<%s>' or `value`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                #                 print('word', word)
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    begin = end
                    break
            sql = "select count(*) from `pkubase` where `prop`='%s' or `prop`='<%s>' or `prop`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    begin = end
                    break
            sql = "select count(*) from `pkuorder` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (
            word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    begin = end
                    break
#             print('end', end)
        begin += 1
#     entity = delete_stopwords(entity)
    return entity

# 根据三元组表pkubase查找实体，逆向最大匹配算法
def search_entity_pkubase_all(line, max_string = False):
    end = len(line)
    entity = []
    new_line = ''
    while end > 0:
        begin = 0
#         print('end', end)
        while (end > begin):
#             print('begin', begin, end = ' ')
            word = line[begin: end]
#             print('w', word)
            sql = "select count(*) from `pkubase` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            begin += 1
            if (data[0][0] > 0):
#                 print(word)
                entity.append(word)
                #                 print('data', data[0:3])
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
                else:
                    continue
            sql = "select count(*) from `pkubase` where `value`='%s' or `value`='<%s>' or `value`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                #                 print(word)
                entity.append(word)
                #                 print('data', data[0:3])
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
                else:
                    continue
            sql = "select count(*) from `pkubase` where `prop`='%s' or `prop`='<%s>' or `prop`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if(data[0][0] > 10):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
                else:
                    continue
            sql = "select count(*) from `pkuorder` where `entry`='%s' or `entry`='<%s>' or `entry`='\"%s\"'" % (word, word, word)
            cur.execute(sql)
            data = cur.fetchall()
            if (data[0][0] > 0):
                entity.append(word)
                if (max_string):  # 是否最大匹配
                    end = begin
                    break
                else:
                    continue
        end -= 1
    entity.reverse()
    return entity

# 根据jieba分词后的个数比较
def select_cut_word_by_jieba_num(entity1, entity2):
    num1 = 0
    num2 = 0
    max1 = 0
    max2 = 0
    for word in entity1:
        num1 += len(jieba.lcut(word))
        if(len(word) > max1):
            max1 = len(word)
    for word in entity2:
        num2 += len(jieba.lcut(word))
        if (len(word) > max2):
            max2 = len(word)
    if(num1 > num2):
        return entity2
    elif num1 < num2:
        return entity1
    elif(max1 >= max2):
        return entity1
    else:
        return entity2


#将正向最大匹配算法和逆向最大匹配算法结合的方法，即相互补充的思想
def combine_forward_backward(entity1, entity2):
    i = 0
    j = 0
    temp1 = ''
    temp1_list = []
    temp2 = ''
    temp2_list = []
    entity_final = []
    while(i < len(entity1)):
        # print('i', i)
        temp1 += entity1[i]
        temp1_list.append(entity1[i])
        i += 1
        if(len(temp1) == len(temp2)):
            if ('\t'.join(temp1_list) == '\t'.join(temp2_list)):
                entity_final.extend(temp1_list)
                temp1 = ''
                temp2 = ''
                temp1_list = []
                temp2_list = []
            else:
#                 print('选择频率高的')
                # entity_final.extend(select_cut_word(temp1_list, temp2_list))
                entity_final.extend(select_cut_word_by_jieba_num(temp1_list, temp2_list))
                # entity_final.extend(select_cut_word_by_jieba(temp1_list, temp2_list))
                temp1 =''
                temp2 =''
                temp1_list = []
                temp2_list = []
        elif(len(temp1) < len(temp2)):
            continue
        else:
            while(j < len(entity2)):
                # print('j', j)
                temp2 += entity2[j]
                temp2_list.append(entity2[j])
                j += 1
                if(len(temp1) == len(temp2)):
                    if('\t'.join(temp1_list) == '\t'.join(temp2_list)):
                        entity_final.extend(temp1_list)
                        temp1 = ''
                        temp2 = ''
                        temp1_list = []
                        temp2_list = []
                    else:
#                         print('选择频率高的')
                        # entity_final.extend(select_cut_word(temp1_list, temp2_list))
                        entity_final.extend(select_cut_word_by_jieba_num(temp1_list, temp2_list))
                        # entity_final.extend(select_cut_word_by_jieba(temp1_list, temp2_list))
                        temp1 =''
                        temp2 = ''
                        temp1_list = []
                        temp2_list = []
                elif(len(temp1) > len(temp2)):
                    continue
                else:
                    break
    return entity_final

# 去掉开头和结尾的'的'
def del_de_extra(word):
    # print('word1', word)
    cut_word = jieba.lcut(word)
    while(len(cut_word) > 0):
        if(cut_word[0] == '的'):
            cut_word[0] = '\n'
        elif(cut_word[-1] == '的'):
            cut_word[-1] = '\n'
        else:
            break
    # print('cut_word', cut_word)
    new_word = '\n'.join(cut_word)
    # print('new_word', new_word)
    # word = new_word.replace('的\n', '\n')
    # word = word.strip('的')
    word = ''.join(cut_word)
    # word = word.replace('\n', '')
    # print('word2', word)
    return word

# 去除开始和结尾处的标点
def delete_punc_extra(line):
    punc = ['《', '》', '\"', '\'', '<', '>', '？', '?', ',', '，', '：']
    # punc = []
    for item in punc:
        line = line.strip(item)
    return line
# # 去除开始和结尾处的标点
# def delete_punc_outside(line):
#     punc = ['《', '》', '\"', '\'', '<', '>', '？', '?', ',', '，', '：']
#     # punc = []
#     for item in punc:
#         line = line.strip(item)
#     return line

# 先预过一遍，得到初步结果，然后去除‘的’
def pre_cut(line_with_punc):
    entity1 = search_entity_pkubase1(line_with_punc)
    entity2 = search_entity_pkubase2(line_with_punc)
    relation = combine_forward_backward(entity1, entity2)
#     relation = entity1
    i = 0
    while i < len(relation):
        relation[i] = del_de_extra(relation[i])
#         relation[i] = delete_punc_extra(relation[i])
        i += 1
    return ''.join(relation)

with open("../data/test.json",'r')as f:
    all_test_data = json.load(f)
    questions = all_test_data[1]
    mentions = all_test_data[6]

# 测试集切分keywords
i = 0
entity_max_match = []
entity_all = []
while (i < len(questions)):
    relation = []
    
    line_with_punc = questions[i]
    #line_with_punc = pre_cut(line_with_punc)
    line = line_with_punc
    print('line:', i, line)
    entity2 = search_entity_pkubase_forward(line)#正向最大匹配分词
    #entity_all.append(entity2)
    entity1 = search_entity_pkubase_backward(line)#逆向最大匹配分词
    #entity_all.append(entity1)
#     print('entity1', entity1)
#     print('entity2', entity2)
    relation = combine_forward_backward(entity1, entity2)#将正向和逆向结合，得到相对更合适的分词结果
    entity = []
    for word in relation:
        if(word not in entity and len(word) > 1):
             entity.append(word)
    entity_max_match.append(entity)
    
    entitys = search_entity_pkubase_all(line_with_punc, False)#得到暴力搜索得到的分词结果
    entity = []
    for word in entitys:
        if(word not in entity and len(word) > 1):
            entity.append(word)
    entity_all.append(entity)
    
    i += 1


fn_out = 'data/questions_ws.txt'
fp_out = open(fn_out, 'w', encoding='utf-8')
lines = questions
i = 0
while (i < len(lines)):
    line_with_punc = lines[i]
    fp_out.write(line_with_punc + '\n')
    fp_out.write('\t'.join(entity_max_match[i]) + '\n')
    fp_out.write('\t'.join(entity_all[i]) + '\n')
    i += 1
fp_out.close()
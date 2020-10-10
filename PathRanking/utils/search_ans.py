from ans_tools import *
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='search answer')
    parser.add_argument("--data_dir", default='../../PathRanking/model/saved_rich', type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    # 原始数据：问题答案所有信息等
    fn1 = '../../data/test_format.json'
    with open(fn1,'r')as f1:
        test_all_data=json.load(f1)
        questions = test_all_data['questions']
        answers = test_all_data['answers']
        qa_data = []
        for q,a in zip(questions,answers):
            qa_data.append((q,a))
    print('all qa data number:',len(qa_data))
    fn_out = os.path.join(data_dir, 'one_hop_best_path_ws_ent.json')
    answer=one_hop_path2ans(fn_out)
    all_p,all_r,all_f = 0,0,0
    num_one_hop=0
    for line in qa_data:
        (q,a)= line
        if q in answer.keys():
            predict=answer[q]
            one_p,one_r,one_f=one_value(predict,a)
            if one_f <0.9:
                pass
                #print(q,a,predict)
            num_one_hop+=1
            all_f+=one_f
    print("one_hop F1 value %f/%d=%f"%(all_f,num_one_hop,all_f/num_one_hop))

    fns = ['left_best_path_ws_ent.json','two_hop_best_path_ws_ent.json']
    for i, fn in enumerate(fns):
        fns[i] = os.path.join(data_dir,fn)
    answer=two_hop_path2ans([fns[0]])

    all_p,all_r,all_f = 0,0,0
    num_one_hop=0
    for line in qa_data:
        (q,a)= line
        if q in answer.keys():
            predict=answer[q]
            one_p,one_r,one_f=one_value(predict,a)
            if one_f <0.9:
                pass
                #print(q,a,predict)
            num_one_hop+=1
            all_f+=one_f
    print("two_hop F1 value %f/%d=%f"%(all_f,num_one_hop,all_f/num_one_hop))

    answer=two_hop_path2ans([fns[1]])

    all_p,all_r,all_f = 0,0,0
    num_one_hop=0
    for line in qa_data:
        (q,a)= line
        if q in answer.keys():
            predict=answer[q]
            one_p,one_r,one_f=one_value(predict,a)
            if one_f <0.9:
                pass
                #print(q,a,predict)
            num_one_hop+=1
            all_f+=one_f
    print("two_hop F1 value %f/%d=%f"%(all_f,num_one_hop,all_f/num_one_hop))

    fn_out = os.path.join(data_dir,'multi_constraint_best_path_ws_ent.json')
    answer=multi_path2ans([fn_out])
    all_p,all_r,all_f = 0,0,0
    num_one_hop=0
    for line in qa_data:
        (q,a)= line
        if q in answer.keys():
            predict=answer[q]
            one_p,one_r,one_f=one_value(predict,a)
            if one_f <0.9:
                pass
                #print(q,a,predict)
            num_one_hop+=1
            all_f+=one_f
    print("multi constraint F1 value %f/%d=%f"%(all_f,num_one_hop,all_f/num_one_hop))
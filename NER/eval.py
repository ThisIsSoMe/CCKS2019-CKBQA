def seq_eval(data, pred, gold, mask, recover):
    pred_list = []
    gold_list = []
    pred = pred[recover]
    gold = gold[recover]
    mask = mask[recover]
    batch_size = gold.size(0)
    seq_len = gold.size(1)
    pred_tag = pred.cpu().data.numpy()
    gold_tag = gold.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    for idx in range(batch_size):
        pred = [data.label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [data.label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_list.append(pred)
        gold_list.append(gold)
    return gold_list, pred_list

def bert_eval(data, pred, gold, mask):
    pred_list = []
    gold_list = []
    batch_size = gold.size(0)
    seq_len = gold.size(1)
    pred_tag = pred.cpu().data.numpy()
    gold_tag = gold.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    for idx in range(batch_size):
        pred = [data.label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [data.label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_list.append(pred)
        gold_list.append(gold)
    return gold_list, pred_list


def get_ner_measure(pred, gold, scheme):
    sen_num = len(pred)
    predict_num = correct_num = gold_num = 0
    for idx in range(sen_num):
        if scheme == "BIO":
            gold_entity = get_entity(gold[idx])
            pred_entity = get_entity(pred[idx])
            predict_num += len(pred_entity)
            gold_num += len(gold_entity)
            correct_num += len(list(set(gold_entity).intersection(set(pred_entity))))
        elif scheme == "BIOES":  # "BMES"
            gold_entity = get_BIOES_entity(gold[idx])
            pred_entity = get_BIOES_entity(pred[idx])
            predict_num += len(pred_entity)
            gold_num += len(gold_entity)
            correct_num += len(list(set(gold_entity).intersection(set(pred_entity))))
        else:
            raise RuntimeError("Scheme Error")
    return predict_num, correct_num, gold_num

def get_BIOES_entity(label_list):
    sen_len = len(label_list)
    entity_list = []
    entity = None
    entity_index = None
    for idx, current in enumerate(label_list):
        if "B-" in current:
            if entity is not None:
                entity_list.append("[" + entity_index + "]" + entity)
            entity = current.split("-")[1]
            entity_index = str(idx)
        elif "S-" in current:
            if entity is not None:
                entity_list.append("[" + entity_index + "]" + entity)
            entity = current.split("-")[1]
            entity_index = str(idx)
        elif "I-" in current or "E-" in current:
            if entity is not None:
                entity_index += str(idx)
            else:
                # print('single I start')
                continue
                entity = current.split("-")[1]
                entity_index = str(idx)
        elif "O" in current:
            if entity is not None:
                entity_list.append("[" + entity_index + "]" + entity)
                entity = None
                entity_index = None
        else:
            print("Label Error. current:{}".format(current))
    if entity is not None:
        entity_list.append("[" + entity_index + "]" + entity)
    return entity_list

def get_entity(label_list):
    entity_list = []
    entity = None
    entity_index = None
    for idx, current in enumerate(label_list):
        if "B-" in current:
            if entity is not None:
                entity_list.append("[" + entity_index + "]" + entity)
            entity = current.split("-")[1]
            entity_index = str(idx)
        elif "I-" in current:
            if entity is not None:
                entity_index += str(idx)
            else:
                #print('single I start')
                continue
                entity = current.split("-")[1]
                entity_index = str(idx)
        else:
            # if current != 'O':
            #     print(current)
            if entity is not None:
                entity_list.append("[" + entity_index + "]" + entity)
                entity = None
                entity_index = None
    if entity is not None:
        entity_list.append("[" + entity_index + "]" + entity)
    return entity_list


def output_result(texts,pred_list,result_dir,info):
    with open(result_dir+'result_'+info,'w',encoding='utf-8') as fout:
        for idx,text in enumerate(texts):
            for idy,t in enumerate(text[0]):
                fout.write(t+'\t'+pred_list[idx][idy]+'\n')
            fout.write('\n')

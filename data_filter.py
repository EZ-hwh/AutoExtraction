from Environment import ExtractionEnv
from transformers import BertTokenizerFast
from tqdm import tqdm
from collections import Counter
import json

dname = 'SKE'

if dname in ['WebNLG','NYT']:
    plm = 'bert-base-cased'
else:
    plm = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizerFast.from_pretrained(plm)

def seq2dict(predict_list, slot_list):
    pre_list = []
    pre = {}
    if dname in ['WebNLG','NYT']:
        for k in predict_list.keys():
            pre = {}
            c = Counter(k)
            if c[':'] == 0:
                gt = predict_list[k]
            if c[';'] == len(slot_list):
                predict_word_offset = []
                for slot in slot_list:
                    predict_word_offset.append((k.index('; ' + slot+':'), len(slot) + 1, slot))
                predict_word_offset.sort()
                for index, offset in enumerate(predict_word_offset):
                    s, l, slot = offset
                    vs = s + 2 + l
                    if index != len(slot_list) - 1:
                        ve = predict_word_offset[index+1][0]
                    else:
                        ve = len(k)
                    #if k[vs:ve] != '[None]':
                    pre[slot] = k[vs:ve]
                pre_list.append(pre)
    else:
        for k in predict_list.keys():
            pre = {}
            c = Counter(k)
            if c['；'] == 0:
                gt = predict_list[k]
            if c['；'] == len(slot_list):
                predict_word_offset = []
                for slot in slot_list:
                    predict_word_offset.append((k.index('； ' + slot+'：'), len(slot) + 1, slot))
                predict_word_offset.sort()
                for index, offset in enumerate(predict_word_offset):
                    s, l, slot = offset
                    vs = s + 2 + l
                    if index != len(slot_list) - 1:
                        ve = predict_word_offset[index+1][0]
                    else:
                        ve = len(k)
                    #if k[vs:ve] != '[None]':
                    pre[slot] = k[vs:ve]
                pre_list.append(pre)
    return pre_list

if dname == 'WebNLG':
    env1 = ExtractionEnv(plm='bert-base-cased', 
                    extraction_model_weight='weight/version1/gp_webnlg.pt', 
                    tokenizer=tokenizer, 
                    data_path='dataset/WebNLG/test.json',
                    dataset='WebNLG',
                    mode='test',
                    lang='en')
    env2 = ExtractionEnv(plm='bert-base-cased', 
                    extraction_model_weight='weight/version1/gp_webnlg.pt', 
                    tokenizer=tokenizer, 
                    data_path='dataset/WebNLG/test.json',
                    dataset='WebNLG',
                    mode='test',
                    lang='en')
elif dname == 'DuEE1.0':
    env1 = ExtractionEnv(plm=plm, 
                    extraction_model_weight='weight/version1/gp_duee.pt', 
                    tokenizer=tokenizer, 
                    data_path='dataset/DuEE1.0/duee_dev.json', 
                    dataset='DuEE1.0',
                    mode='test',
                    lang='zh')
    env2 = ExtractionEnv(plm=plm, 
                    extraction_model_weight='weight/version1/gp_duee.pt', 
                    tokenizer=tokenizer, 
                    data_path='dataset/DuEE1.0/duee_dev.json', 
                    dataset='DuEE1.0',
                    mode='test',
                    lang='zh')
elif dname == 'DuIE2.0':
    env1 = ExtractionEnv(plm=plm, 
                    extraction_model_weight='weight/vertsion1/gp_duie.pt',
                    tokenizer=tokenizer, 
                    data_path='dataset/DuIE2.0/duie_dev.json', 
                    dataset='DuIE2.0',
                    mode='test',
                    lang='zh')
    env2 = ExtractionEnv(plm=plm, 
                    extraction_model_weight='weight/version1/gp_duie.pt',
                    tokenizer=tokenizer, 
                    data_path='dataset/DuIE2.0/duie_dev.json', 
                    dataset='DuIE2.0',
                    mode='test',
                    lang='zh')
elif dname == 'DuEE-fin':
    env1 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_duee_fin.pt',
                    tokenizer=tokenizer,
                    data_path='dataset/DuEE-fin/duee_fin_dev_new.json',
                    dataset='DuEE-fin',
                    mode='test',
                    lang='zh')
    env2 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_duee_fin.pt',
                    tokenizer=tokenizer,
                    data_path='dataset/DuEE-fin/duee_fin_dev_new.json',
                    dataset='DuEE-fin',
                    mode='test',
                    lang='zh')
elif dname == 'HacRED':
    env1 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_hacred.pt',
                    tokenizer=tokenizer,
                    data_path='dataset/HacRED/new_test.json',
                    dataset='HacRED',
                    mode='test',
                    lang='zh')
    env2 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_hacred.pt',
                    tokenizer=tokenizer,
                    data_path='dataset/HacRED/new_test.json',
                    dataset='HacRED',
                    mode='test',
                    lang='zh')
elif dname == 'NYT':
    env1 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_nyt.pt',
                    tokenizer=tokenizer,
                    data_path='data/NYT/new_test.json',
                    dataset='NYT',
                    mode='test',
                    lang='en')
    env2 = ExtractionEnv(plm=plm,
                    extraction_model_weight='weight/version1/gp_nyt.pt',
                    tokenizer=tokenizer,
                    data_path='data/NYT/new_test.json',
                    dataset='NYT',
                    mode='test',
                    lang='en')
elif dname == 'SKE':
    env1 = ExtractionEnv(plm=plm,
                    extraction_model_weight=f'weight/version1/gp_ske.pt',
                    tokenizer=tokenizer,
                    data_path='data/ske2019/new_test.json',
                    dataset='SKE',
                    mode='test',
                    lang='zh')
    env2 = ExtractionEnv(plm=plm,
                    extraction_model_weight=f'weight/version1/gp_ske.pt',
                    tokenizer=tokenizer,
                    data_path='data/ske2019/new_test.json',
                    dataset='SKE',
                    mode='test',
                    lang='zh')

print('Begin Filtering!')

tot = 0

overlap_text = []

for i_episode in tqdm(range(env1.dataset_len),desc='Extract through RL agent'): 
    state_list, _, _ = env1.reset()
    slot_list = state_list[0][2]
    # Restrict slot_num
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, bert_input, choices = state 
            action = 0
            next_state_list, reward, done = env1.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break

    state_list, _, _ = env2.reset()
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, bert_input, choices = state 
            action = -1
            next_state_list, reward, done = env2.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break
    #print(env.return_cond().keys())
    predict_list1 = env1.return_cond()
    predict_list2 = env2.return_cond()
    pred1 = seq2dict(predict_list1, slot_list)
    pred2 = seq2dict(predict_list2, slot_list)
    if pred1 == pred2:
        continue
    overlap_text.append(env1.text)
    #print(env1.text)

#print(len(overlap_text))

label_datas = []
if dname == 'SKE':
    with open(f'data/ske2019/new_test.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label_datas.append(json.loads(line))

""" if dname == 'WebNLG':
    with open(f'dataset/{dname}/test.json', 'r', encoding='utf-8') as f:
        label_datas = json.loads(f.read())
else:
    label_datas = []
    with open(f'dataset/{dname}/duee_fin_dev_new.json', 'r', encoding='utf-8') as f:
        for line in f.readlines():
           label_datas.append(json.loads(line)) """

filter_data = []
for data in label_datas:
    if data['sentText'] in overlap_text:
        filter_data.append(data)

if dname == 'SKE':
    with open(f'data/ske2019/test_filter.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(filter_data))

""" if dname == 'WebNLG':
    with open(f'dataset/{dname}/test_filter.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(filter_data))
else:
    with open(f'dataset/{dname}/duee_fin_dev_new_filter.json', 'w', encoding='utf-8') as f:
        for data in filter_data:
            f.write(json.dumps(data, ensure_ascii=False)+'\n') """
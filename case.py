import torch
import json
from tqdm import tqdm
from model import RCModel
from transformers import BertTokenizerFast
from Environment import ExtractionEnv
from RL_utils.dqn import DQN
from collections import Counter

#Params
rc_threshold = 0.5
#plm = 'bert-base-cased'
plm = 'hfl/chinese-roberta-wwm-ext'
dataset = 'DuEE1.0'
lang = 'zh'
model_name = 'duee'

# load data and rel_map

# load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(plm)
# load model
#rc = RCModel(plm,len(rel_map)).cuda()
#rc.load_state_dict(torch.load(f'/mnt/huangwenhao/TripleFilter/{dataset}/rc.pt'))
agent1 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
agent1.load_weight(f'weight/{model_name}/v2_rl_1.pt')
agent2 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
agent2.load_weight(f'weight/{model_name}/v2_rl_2.pt')

# load environment
env = ExtractionEnv(plm=plm,
                    extraction_model_weight=f'weight/version1/gp_duee.pt',
                    tokenizer=tokenizer,
                    data_path=f'data/{dataset}/duee_train.json',
                    dataset=dataset,
                    mode='test',
                    lang=lang)

def spo2text_zh(spo): return spo['relation'] + '|' + spo['头实体'] + '|' + spo['尾实体']
def spo2text_en(spo): return spo['relation'] + '|' + spo['subject'] + '|' + spo['object']
def spo2text_gt(spo): return spo['label'] + '|' + spo['em1Text'] + '|' + spo['em2Text']

fout = open(f'case/DuEE.txt', 'w', encoding='utf-8')

# Predict Process
for i_episode in tqdm(range(env.dataset_len),desc='Extract through RL agent'): 
    state_list, _, _ = env.reset()
    # Restrict slot_num
    ori_cond, text, slot_list = state_list[0]
    slot_num = len(slot_list)
    if slot_num <= 2:
        continue
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, text, choices = state
            # Auto extraction
            action1 = agent1.select_action(cond, text, choices)
            action2 = agent2.select_action(cond, text, choices)
            action = torch.argmax(action1 + action2)
            next_state_list, reward, done = env.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break
    pre_list = []
    predict_list = env.return_cond()
    pred_text_list = []
    for k in predict_list.keys():
        """ if '[None]' in k:
            continue """
        c = Counter(k)
        if (c['；'] == 0 and lang == 'zh') or (c[';'] == 0 and lang == 'en'):
            gt = predict_list[k]
        if (c['；'] == slot_num and lang == 'zh') or (c[';'] == slot_num and lang == 'en'):
            pred_text_list.append(k)
            pre = {'relation': ori_cond}
            predict_word_offset = []
            for slot in slot_list:
                if lang == 'zh':
                    predict_word_offset.append((k.index('； ' + slot+'：'), len(slot) + 1, slot))
                elif lang == 'en':
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
    print('#'*50, file=fout)
    print(text, file=fout)
    print(pre_list, file=fout)
    print(gt, file=fout)
    print(pred_text_list, file=fout)

fout.close()
import torch
import json
from tqdm import tqdm
from model import RCModel
from transformers import BertTokenizerFast
from Environment import ExtractionEnv
from RL_utils.dqn import DQN
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plm', type=str, default='bert-base-cased', choices=['bert-base-cased', 'hfl/chinese-roberta-wwm-ext'], help='Pretrained Language Model')
parser.add_argument('--dataset', type=str, default='NYT10-HRL', help='Dataset')
parser.add_argument('--lang', type=str, default='en', help='Language')
parser.add_argument('--model_name', type=str, default='nyt10-hrl', help='Model Name')
args = parser.parse_args()

#Params
rc_threshold = 0.6
plm = args.plm
dataset = args.dataset
lang = args.lang
model_name = args.model_name

# load data and rel_map
with open(f'data/{args.dataset}/rel2id.json', 'r', encoding='utf-8') as f:
    rel_map = json.load(f)
rev_rel_map = {v: k for k, v in rel_map.items()}

with open(f'data/{dataset}/new_test.json', 'r', encoding='utf-8') as f:
    datas = []
    for line in f.readlines():
        datas.append(json.loads(line))

# load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(plm)
# load model
rc = RCModel(plm,len(rel_map)).cuda()
rc.load_state_dict(torch.load(f'weight/{model_name}/rc.pt'))
agent1 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
agent1.load_weight(f'weight/{model_name}/v2_rl_1.pt')
agent2 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
agent2.load_weight(f'weight/{model_name}/v2_rl_2.pt')

# load environment
env = ExtractionEnv(plm=plm,
                    extraction_model_weight=f'weight/version1/gp_nyt10.pt',
                    tokenizer=tokenizer,
                    data_path=f'data/{dataset}/new_test.json',
                    dataset=dataset,
                    mode='test',
                    lang=lang)

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)

    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

    # 为了绘制PR curve打印到文件上
    def compute_and_record(self, fout):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        fout.write(pstr+'\n')
        return (prec, reca, f1)

def ext_with_env(text, ori_cond, choices, lang):
    state_list, _, _ = env.reset_with_input(text, ori_cond, choices)
    slot_list = state_list[0][2]
    slot_num = len(slot_list)
    ep_reward = 0
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, text, choices = state
            action1 = agent1.select_action(cond, text, choices)
            action2 = agent2.select_action(cond, text, choices)
            action = torch.argmax(action1 + action2) 
            # action = 0
            next_state_list, reward, done = env.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break
    #print(env.return_cond().keys())
    pre_list = []
    predict_list = env.return_cond()
    for k in predict_list.keys():
        if '[None]' in k:
            continue
        c = Counter(k)
        if (c['；'] == 0 and lang == 'zh') or (c[';'] == 0 and lang == 'en'):
            gt = predict_list[k]
        if (c['；'] == slot_num and lang == 'zh') or (c[';'] == slot_num and lang == 'en'):
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
    return pre_list

def spo2text_zh(spo): return spo['relation'] + '|' + spo['头实体'] + '|' + spo['尾实体']
def spo2text_en(spo): return spo['relation'] + '|' + spo['subject'] + '|' + spo['object']
def spo2text_gt(spo): return spo['label'] + '|' + spo['em1Text'] + '|' + spo['em2Text']

# Predict Process
f1 = MetricF1()
for data in tqdm(datas):
    output = tokenizer(data['sentText'], return_token_type_ids=True, return_offsets_mapping=True)
    input_ids = output['input_ids'][:512]
    token_type_ids = output['token_type_ids'][:512]
    labels = torch.zeros(len(rel_map)).int()
    input_ids = torch.IntTensor(input_ids).unsqueeze(0).cuda()
    token_type_ids = torch.IntTensor(token_type_ids).unsqueeze(0).cuda()
    labels = labels.unsqueeze(0).cuda()
    output = rc(input_ids).squeeze()
    pred = set()
    for i, v in enumerate(output):
        if v > rc_threshold: 
            if lang == 'zh':
                predict = ext_with_env(data['sentText'], rev_rel_map[i], ['头实体','尾实体'], lang)
                for spo in predict:
                    pred.add(spo2text_zh(spo))
            elif lang == 'en':
                predict = ext_with_env(data['sentText'], rev_rel_map[i], ['subject','object'], lang)
                for spo in predict:
                    pred.add(spo2text_en(spo))
            #print(predict)
            
    gold = set([spo2text_gt(spo) for spo in data['relationMentions']])
    f1.append(pred, gold)

f1.compute()
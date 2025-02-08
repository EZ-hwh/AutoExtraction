import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import permutations
import copy
import random

class DuEE_finDataset(Dataset):
    '''
    传统的抽取式数据构造方案
    '''
    def __init__(self, data_path, tokenizer, data_type='extraction', data_split=1):
        self.data_split = data_split
        self._load_schema()
        self._load_dataset(data_path)
        self._process_data()
        self.data_type = data_type
        self.tokenizer = tokenizer
        if data_type == 'extraction':
            self._gen_data_for_extraction_model()
        elif data_type == 'rl':
            self._gen_data_for_rl_model()

    def _load_schema(self):
        self.schema = {}
        with open('data/DuEE-fin/duee_fin_event_schema.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                res = json.loads(line)
                self.schema[res['event_type']] = [item['role'] for item in res['role_list']]

    def _process_data(self):
        '''
        对数据进行初始的预处理，主要是将触发词作为正常的论元进行抽取。
        '''
        self.datas = []
        for data in self.label_datas:
            eve = {}
            if 'event_list' not in data.keys():
                continue
            for event in data['event_list']:
                if event['event_type'] not in eve.keys():
                    eve[event['event_type']] = []
                tmp_event = {
                    'event_type': event['event_type']
                }
                tmp_event['arguments'] = {
                    '触发词': [event['trigger']]
                }
                for argument in event['arguments']:
                    if argument['role'] not in tmp_event['arguments'].keys():
                        tmp_event['arguments'][argument['role']] = []
                    tmp_event['arguments'][argument['role']].append(argument['argument'])

                event_list = [{}]
                for argument in tmp_event['arguments'].keys():
                    new_event_list = []
                    for el in event_list:
                        for entity in tmp_event['arguments'][argument]:
                            tmp_el = copy.deepcopy(el)
                            tmp_el[argument] = entity
                            new_event_list.append(tmp_el)
                    event_list = new_event_list
                eve[event['event_type']].extend(event_list)
            self.datas.append({
                    'text': data['text'],
                    'event_list': eve
                })

    def _gen_data_for_extraction_model(self):
        new_data = []
        for data in tqdm(self.datas[0:],desc='Process data for extraction model'):
            '''构造数据：
            1. 直接预测第一个（头尾）实体，无额外信息
            2. 根据第一个（头尾）实体信息，预测后一个信息
            '''
            text = data['text']
            #print(text)
            text_len = len(self.tokenizer.tokenize(text))
            for event_type, event_list in data['event_list'].items():
                ## 全排列构造所有可能的抽取顺序
                for event in event_list:
                    event_keys = [i for i in event.keys()]
                    event_keys = self.schema[event_type]
                    if len(event_keys) >= 5:
                        for _ in range(10):
                            random.shuffle(event_keys)
                            cond = f'{event_type}；'
                            for argument_key in event_keys:
                                new_cond = cond
                                while len(self.tokenizer.tokenize(f'{new_cond} {argument_key}：')) + text_len + 3 > 512:
                                    new_cond = '；'.join(new_cond.split('；')[:-1])
                                cond = f'{new_cond} {argument_key}：'
                                if argument_key in event.keys():
                                    new_data.append((cond, text, event[argument_key]))
                                    cond += f'{event[argument_key]}；'
                                else:
                                    new_data.append((cond, text, '[None]'))
                                    cond += f'[None]；'
                    else:
                        for event_permutation in permutations(event_keys):
                            # 待抽取元素进行全排列
                            cond = f'{event_type}；'
                            for argument_key in event_permutation:
                                new_cond = cond
                                while len(self.tokenizer.tokenize(f'{new_cond} {argument_key}：')) + text_len + 3 > 512:
                                    new_cond = '；'.join(new_cond.split('；')[:-1])
                                cond = f'{new_cond} {argument_key}：'
                                if argument_key in event.keys():
                                    new_data.append((cond, text, event[argument_key]))
                                    cond += f'{event[argument_key]}；'
                                else:
                                    new_data.append((cond, text, '[None]'))
                                    cond += f'[None]；'

                    """ event_keys = list(event['arguments'].keys())
                    #print(event_keys)
                    if len(event_keys) >= 5:
                        for _ in range(10):
                            random.shuffle(event_keys)
                            cond_list = [f'{event_type}；']

                            for argument_key in event_keys:
                                new_cond_list = []
                                for cond in cond_list:
                                    new_cond = cond
                                    while len(self.tokenizer.tokenize(f'{new_cond} {argument_key}：')) + text_len + 3 > 512:
                                        new_cond = '；'.join(new_cond.split('；')[:-1])
                                        
                                    new_data.append((f'{new_cond} {argument_key}：', text, event["arguments"][argument_key]))
                                    for arguments in event['arguments'][argument_key]:
                                        new_cond_list.append(f'{new_cond} {arguments["role"]}：{arguments["argument"]}；')
                                cond_list = new_cond_list
                    else:
                        for event_permutation in permutations(event_keys):
                            # 待抽取元素进行全排列
                            cond_list = [f'{event_type}；']

                            for argument_key in event_permutation:
                                new_cond_list = []
                                for cond in cond_list:
                                    new_cond = cond
                                    while len(self.tokenizer.tokenize(f'{new_cond} {argument_key}：')) + text_len + 3 > 512:
                                        new_cond = '；'.join(new_cond.split('；')[:-1])
                                        
                                    new_data.append((f'{new_cond} {argument_key}：', text, event["arguments"][argument_key]))
                                    for arguments in event['arguments'][argument_key]:
                                        new_cond_list.append(f'{new_cond} {arguments["role"]}：{arguments["argument"]}；')
                                cond_list = new_cond_list """
                            
        new_data.sort(key=lambda item:(item[0],item[1]))
        self.datas = []
        old_data = None
        for data in new_data:
            if old_data and (data[0], data[1]) == (old_data[0], old_data[1]):
                self.datas[-1][2].append(data[2])
            else:
                self.datas.append((data[0],data[1],[data[2]]))
            old_data = data

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            '''构造数据：
            1. 把待抽取的句子和Ground Truth的一个关系放在一起，针对关系已知的句子进行要素抽取
            2. 一个Example就是一个抽取的环境，简单而言就是一个（Text, Predicate）对。
            '''
            for event in data['event_list'].keys():
                if len(data['event_list'][event]) < 0: continue
                new_data.append((data['text'], event, data['event_list'][event]))
        self.datas = new_data

    def _recognize(self, text, logits, offset_mapping, threshold = 0):
        import numpy as np
        '''
        根据GlobalPointer模型的输出进行抽取结果的识别，返回形式：
            [(entity, scores),...]
        '''
        scores = logits.squeeze().cpu() # Size: [seqlen, seqlen]
        entities = []
        for start, end in zip(*np.where(scores > threshold)):
            print(start, end)
            entities.append(
                (text[offset_mapping[start][0]: offset_mapping[end][1]], scores[start,end])
            )
        if entities:
            entities.sort(key=lambda x:x[-1], reverse=True)
        
        return entities

    def _find_pos(self, entity, input_ids):
        #print(entity)
        entity_tokens = self.tokenizer.tokenize(entity)
        entity_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
        ret = []
        for index in range(len(input_ids)):
            if entity_ids[0] == input_ids[index]:
                flag = True
                for i in range(len(entity_ids)):
                    if index + i >= len(input_ids) or input_ids[index+i] != entity_ids[i]:
                        flag=False
                        break
                if flag:
                    ret.append((index,index+len(entity_ids)))
        return ret

    def example_generation(self, text, cond, slot_list):
        '''
        将带抽取的要素设置为Prompt
        '''
        #print(cond, text, slot_list)
        output = self.tokenizer(cond, text, return_token_type_ids=True, return_offsets_mapping=True)
        input_ids = output['input_ids']
        token_type_ids = output['token_type_ids']
        offset_mapping = output['offset_mapping']

        labels = torch.zeros(len(input_ids), len(input_ids)).int()
        for slot in slot_list:
            slot_spans = self._find_pos(slot, input_ids)
            for s,e in slot_spans:
                if token_type_ids[s] * token_type_ids[e] == 1:
                    labels[s][e-1] = 1
        return torch.IntTensor(input_ids), torch.IntTensor(token_type_ids), labels

    def _load_dataset(self, data_path):
        self.label_datas = []
        if 'train' in data_path:
            with open(data_path[:-5]+f'{self.data_split}.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.label_datas.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.label_datas.append(json.loads(line))

        """ dataset_len = len(self.label_datas)
        if self.data_split == 1:
            self.label_datas = self.label_datas[:dataset_len // 2]
        else:
            self.label_datas = self.label_datas[dataset_len // 2:] """

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        examples = self.datas[index]
        if self.data_type == 'extraction':
            cond, text, slot_list = examples
            return self.example_generation(text, cond, slot_list)
        elif self.data_type == 'rl':
            return examples

    def collate_fn_cuda(self, batch):
        bz = len(batch)
        maxlen = max([len(item[0]) for item in batch])
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = torch.zeros(bz,1,maxlen,maxlen)

        for index, (input_ids, token_type_ids, labels) in enumerate(batch):
            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            seqlen = len(input_ids)
            #print(labels.size())
            #print(seqlen)
            batch_labels[index][0][:seqlen,:seqlen] = labels

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type_ids, batch_first=True)

        return [
            batch_input_ids.cuda(),
            batch_token_type_ids.cuda(),
            batch_labels.cuda()
        ]
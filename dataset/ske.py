import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import permutations
import copy
import random

class SKEDataset(Dataset):
    def __init__(self, data_path, tokenizer, data_type='extraction', data_split=1):
        self.data_split = data_split
        self._load_dataset(data_path)
        self._process_data()
        self.data_type = data_type
        self.tokenizer = tokenizer
        if data_type == 'extraction':
            self._gen_data_for_extraction_model()
        elif data_type == 'rl':
            self._gen_data_for_rl_model()

    def _process_data(self):
        '''
        对数据进行初始的预处理，主要是将每句话的关系进行归类。
        '''
        self.datas = []
        for data in self.label_datas:
            rel = {}
            for relation in data['relationMentions']:
                if relation['label'] not in rel.keys():
                    rel[relation['label']] = []
                new_relation = {
                    '头实体': relation['em1Text'],
                    '关系': relation['label'],
                    '尾实体': relation['em2Text']
                }
                rel[relation['label']].append(new_relation)
            self.datas.append({
                'text': data['sentText'],
                'relation_list': rel
            })

    def _gen_data_for_extraction_model(self):
        new_data = []
        for data in tqdm(self.datas,desc='Process data for extraction model'):
            '''构造数据：
            1. 直接预测第一个（头尾）实体，无额外信息
            2. 根据第一个（头尾）实体信息，预测后一个信息
            '''
            for relation in data['relation_list'].keys():
                subjects = [item['头实体'] for item in data['relation_list'][relation]]
                objects = [item['尾实体'] for item in data['relation_list'][relation]]
                # subjects = list(set([(item['subject'],tuple(item['subj_char_span'])) for item in data['relation_list'][relation]]))
                #   objects = list(set([(item['object'],tuple(item['obj_char_span'])) for item in data['relation_list'][relation]]))
                # 构造第一类数据
                new_data.append((f'{relation}； 头实体：', data['text'], subjects))
                new_data.append((f'{relation}； 尾实体：', data['text'], objects))

                # 构造第二类数据
                for subject in subjects:
                    slot_list = []
                    for relation_tuple in data['relation_list'][relation]:
                        if subject == relation_tuple['头实体']:
                            slot_list.append(relation_tuple['尾实体'])
                    new_data.append((f'{relation}； 头实体： {subject}； 尾实体：', data['text'], slot_list))

                for object in objects:
                    slot_list = []
                    for relation_tuple in data['relation_list'][relation]:
                        if object == relation_tuple['尾实体']:
                            slot_list.append(relation_tuple['头实体'])
                    new_data.append((f'{relation}； 尾实体： {object}； 头实体：', data['text'], slot_list))
        self.datas = new_data

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            '''构造数据：
            1. 把待抽取的句子和Ground Truth的一个关系放在一起，针对关系已知的句子进行要素抽取
            2. 一个Example就是一个抽取的环境，简单而言就是一个（Text, Predicate）对。
            '''
            for relation in data['relation_list'].keys():
                if len(data['relation_list'][relation]) >= 0:
                    new_data.append((data['text'], relation, data['relation_list'][relation]))
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
        output = self.tokenizer(cond, text, return_token_type_ids=True, return_offsets_mapping=True)
        
        input_ids = output['input_ids'][:512]
        token_type_ids = output['token_type_ids'][:512]
        offset_mapping = output['offset_mapping'][:512]
        labels = torch.zeros(len(input_ids), len(input_ids)).int()
        for slot in slot_list:
            slot_spans = self._find_pos(slot, input_ids)
            for s,e in slot_spans:
                if token_type_ids[s] * token_type_ids[e-1] == 1:
                    labels[s][e-1] = 1

        return torch.IntTensor(input_ids), torch.IntTensor(token_type_ids), labels

    def _load_dataset(self, data_path):
        self.label_datas = []
        if 'train' in data_path:
            if self.data_split in [1,2]:
                with open(data_path[:-5]+f'{self.data_split}.json', 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        self.label_datas.append(json.loads(line))
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        self.label_datas.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.label_datas.append(json.loads(line))

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
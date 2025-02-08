import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import permutations
import copy
import random

class WebNLGDataset(Dataset):
    '''
    传统的抽取式数据构造方案
    '''
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
        elif data_type == 'classification':
            self.load_class_mapping()
            self._gen_data_for_classification_model()

    def load_class_mapping(self):
        with open('data/WebNLG/rel2id.json') as f:
            self.class_mapping = json.loads(f.read())

    def _process_data(self):
        '''
        对数据进行初始的预处理，主要是将每句话的关系进行归类。
        '''
        self.datas = []
        for data in self.label_datas:
            rel = {}
            for relation in data['relation_list']:
                if relation['predicate'] not in rel.keys():
                    rel[relation['predicate']] = []
                rel[relation['predicate']].append(relation)
            self.datas.append({
                'text': data['text'],
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
                subjects = list(set([(item['subject'],tuple(item['subj_char_span'])) for item in data['relation_list'][relation]]))
                objects = list(set([(item['object'],tuple(item['obj_char_span'])) for item in data['relation_list'][relation]]))
                # 构造第一类数据
                new_data.append((f'{relation}; subject:', data['text'], subjects))
                new_data.append((f'{relation}; object:', data['text'], objects))

                # 构造第二类数据
                for subject in subjects:
                    slot_list = []
                    for relation_tuple in data['relation_list'][relation]:
                        if subject[0] == relation_tuple['subject'] and subject[1] == tuple(relation_tuple['subj_char_span']):
                            slot_list.append((relation_tuple['object'], relation_tuple['obj_char_span']))
                    new_data.append((f'{relation}; subject: {subject[0]}; object:', data['text'], slot_list))

                for object in objects:
                    slot_list = []
                    for relation_tuple in data['relation_list'][relation]:
                        if object[0] == relation_tuple['object'] and object[1] == tuple(relation_tuple['obj_char_span']):
                            slot_list.append((relation_tuple['subject'], relation_tuple['subj_char_span']))
                    new_data.append((f'{relation}; object: {object[0]}; subject:', data['text'], slot_list))
        self.datas = new_data

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            '''构造数据：
            1. 把待抽取的句子和Ground Truth的一个关系放在一起，针对关系已知的句子进行要素抽取
            2. 一个Example就是一个抽取的环境，简单而言就是一个（Text, Predicate）对。
            '''
            for relation in data['relation_list'].keys():
                #if len(data['relation_list'][relation]) >= 2:
                new_data.append((data['text'], relation, data['relation_list'][relation]))
        self.datas = new_data

    """ def _gen_data_for_classification_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for classification model'):
            if dataset """


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

    def example_generation(self, text, cond, slot_list):
        '''
        将带抽取的要素设置为Prompt
        '''
        output = self.tokenizer(cond, text, return_token_type_ids=True, return_offsets_mapping=True)
        input_ids = output['input_ids']
        token_type_ids = output['token_type_ids']
        offset_mapping = output['offset_mapping']
        #print(offset_mapping)
        #print(token_type_ids)
        labels = torch.zeros(len(input_ids), len(input_ids)).int()
        for slot, slot_span in slot_list:
            s, e = 0, 0
            for index in range(len(input_ids)):
                if token_type_ids[index] == 0 or offset_mapping[index] == (0,0):
                    continue
                if slot_span[0] == offset_mapping[index][0]:
                    s = index
                if slot_span[1] == offset_mapping[index][1]:
                    e = index
                    break
            labels[s][e] = 1

        return torch.IntTensor(input_ids), torch.IntTensor(token_type_ids), labels

    def _load_dataset(self, data_path):
        self.label_datas = []
        if 'train' in data_path:
            if self.data_split in [1,2]:
                with open(data_path[:-5]+f'{self.data_split}.json', 'r', encoding='utf-8') as f:
                    self.label_datas = json.loads(f.read())
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.label_datas = json.loads(f.read())
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.label_datas = json.loads(f.read())

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
            batch_labels[index][0][:seqlen,:seqlen] = labels

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type_ids, batch_first=True)

        return [
            batch_input_ids.cuda(),
            batch_token_type_ids.cuda(),
            batch_labels.cuda()
        ]
from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import permutations
import copy
import random

class WebNLGDataset_gen(Dataset):
    '''
    针对生成式构造的数据版本，目前只是按照一定的顺序进行生成。
    '''
    def __init__(self, data_path, tokenizer):
        self._load_dataset(data_path)
        self.tokenizer = tokenizer

    def _process_data(self):
        pass

    def example_generation(self, text, triples):
        #tokens = self.tokenizer.tokenize(text)
        #input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        #input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        input_ids = self.tokenizer(text)['input_ids']

        output_tokens = []
        for relations in triples:
            output_tokens += self.tokenizer.tokenize(relations['predicate']) + [';']
            output_tokens += self.tokenizer.tokenize(relations['subject']) + [';']
            output_tokens += self.tokenizer.tokenize(relations['object']) + ['|']

        #output_tokens += [self.tokenizer.sep_token] 

        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)

        return input_ids, output_ids

    def _load_dataset(self, data_path):
        self.label_datas = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.label_datas = json.loads(f.read())

    def __len__(self):
        return len(self.label_datas)

    def __getitem__(self, index):
        examples = self.label_datas[index]
        (input_ids, output_ids) = self.example_generation(examples['text'],examples['relation_list'])
        return [
            torch.LongTensor(input_ids),
            torch.LongTensor(output_ids)
        ]

    def collate_fn_cuda(self, batch):
        batch_inputs = []
        batch_outputs = []

        for (input_ids, output_ids) in batch:
            batch_inputs.append(input_ids)
            batch_outputs.append(output_ids)

        batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
        batch_outputs = torch.nn.utils.rnn.pad_sequence(batch_outputs, batch_first=True)

        return [
            batch_inputs.cuda(),
            batch_outputs.cuda()
        ]


if __name__ == '__main__':
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
    #dataset = WebNLGDataset('data/WebNLG/train_data.json', tokenizer)
    #dataset = DuIEDataset('data/DuIE2.0/duie_sample.json', tokenizer)
    dataset = DuEEDataset('data/DuEE1.0/duee_train.json', tokenizer)
    #dataset = HacREDDataset('data/HacRED/new_train.json', tokenizer)
    for i in range(15):
        print('*'* 50)
        print(dataset[i])
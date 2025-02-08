from html import entities
import json
import torch
from dataset.nyt import NYTDataset
from dataset.webnlg import WebNLGDataset
from dataset.duee import DuEEDataset
from dataset.duie import DuIEDataset
from dataset.duee_fin import DuEE_finDataset
from dataset.hacred import HacREDDataset
from dataset.ske import SKEDataset
from model import GlobalPointerModel
import random, math
import copy
import numpy as np
from utils import text_f1

class ExtractionEnv:
    def __init__(self, plm, extraction_model_weight, tokenizer, data_path, dataset='WebNLG', lang='en', mode='train', reward_type='v1', data_split=1):
        self.data = None
        self.state = None
        self.data_split = data_split
        self.extraction_model = GlobalPointerModel(plm).cuda()
        self.extraction_model.load_state_dict(torch.load(extraction_model_weight))
        self.tokenizer = tokenizer
        self.mode = mode
        self.dname = dataset
        self.index = 0
        self.lang = lang
        self.reward_type = reward_type

        if dataset == 'WebNLG':
            self.dataset = WebNLGDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuEE1.0':
            self.dataset = DuEEDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuIE2.0':
            self.dataset = DuIEDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuEE-fin':
            self.dataset = DuEE_finDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)
        elif dataset == 'HacRED':
            self.dataset = HacREDDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'NYT':
            self.dataset = NYTDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'SKE':
            self.dataset = SKEDataset(tokenizer=tokenizer, data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)

    def _example_generation(self, text, cond):
        output = self.tokenizer(cond, text, return_token_type_ids=True, return_offsets_mapping=True)
        input_ids = output['input_ids'][:512]
        token_type_ids = output['token_type_ids'][:512]
        offset_mapping = output['offset_mapping'][:512]
        offset = len([i for i in token_type_ids if i == 0])
        return torch.IntTensor(input_ids).cuda(), torch.IntTensor(token_type_ids).cuda(), offset, offset_mapping

    def _load_schema(self):
        self.schema = {}
        if self.dname == 'DuEE1.0':
            with open('data/DuEE1.0/duee_event_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['event_type']] = [item['role'] for item in res['role_list']]
        elif self.dname == 'DuIE2.0':
            with open('data/DuIE2.0/duie_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['predicate']] = ['头实体-' + res['subject_type']]
                    for role in res['object_type'].keys():
                        if role == '@value':
                            self.schema[res['predicate']].append('尾实体-' + res['object_type'][role])
                        else:
                            self.schema[res['predicate']].append('尾实体-' + res['object_type'][role])
        elif self.dname == 'DuEE-fin':
            with open('data/DuEE-fin/duee_fin_event_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['event_type']] = [item['role'] for item in res['role_list']]

    def sigmoid(self, i):
        return 1 / (math.exp(-i) + 1)

    def score2prob(self, entities):
        '''
        Input: Entity without filtering
        Output: Unduplicated list: [(entity, prob, score)]
        '''
        entities_mention = list(set([e[0] for e in entities]))
        logsum = sum([math.exp(e[1]) for e in entities])
        entities = [(e[0],math.exp(e[1])/logsum, e[1]) for e in entities]
        entities_score = [(name, sum([i[1] for i in entities if i[0] == name]), max([i[2] for i in entities if i[0] == name])) for name in entities_mention]
        return entities_score

    def choice_decision(self, cond, choices, action, step):
        """ print('='* 50)
        print(choices[action]) """

        if self.lang == 'en':
            new_cond = f'{cond}; {choices[action]}:'
        elif self.lang == 'zh':
            new_cond = f'{cond}； {choices[action]}：'
        new_choices = copy.deepcopy(choices)
        del new_choices[action]
        input_ids, token_type_ids, offset, offset_mapping = self._example_generation(self.text, new_cond)
        
        with torch.no_grad():
            logits = self.extraction_model(input_ids.unsqueeze(0), token_type_ids.unsqueeze(0))
            entities_1step = self._recognize(self.text, logits[0,0][offset:,offset:], offset_mapping[offset:])
        #entities_1step = [(i,self.sigmoid(j)) for (i,j) in entities_1step]
        entities_1step = self.score2prob(entities_1step)
        if entities_1step == []:
            entities_1step.append(('[None]',0.9, 3))
        if step == 0:
            return entities_1step
        # print(entities_1step)
        if step == 1:
            prob_num = 0
            for spo in self.spo_list[cond]:
                for entity in entities_1step:
                    if isinstance(spo[choices[action]], tuple):
                        if spo[choices[action]][0] == entity[0]:
                            # prob_num += entity[1] * entity[2]
                            prob_num += entity[2]
                    else:
                        if spo[choices[action]] == entity[0]:
                            prob_num += entity[2]
            return prob_num

        entities_2step = []
        batch_ids, batch_token_types, batch_offset, batch_mapping, batch_key = [], [], [], [], []
        for entity in entities_1step:
            for choice in new_choices:
                if self.lang == 'en':
                    tmp_cond = f'{new_cond}{entity[0]}; {choice}:'
                elif self.lang == 'zh':
                    tmp_cond = f'{new_cond}{entity[0]}； {choice}：'
                input_ids, token_type_ids, offset, offset_mapping = self._example_generation(self.text, tmp_cond)
                batch_ids.append(input_ids)
                batch_token_types.append(token_type_ids)
                batch_offset.append(offset)
                batch_mapping.append(offset_mapping)
                batch_key.append((choices[action],choice))

        for minibatch_index in range((len(batch_ids) - 1) // 16 + 1):
            input_ids = torch.nn.utils.rnn.pad_sequence(batch_ids[minibatch_index * 16: (minibatch_index + 1) * 16], batch_first=True).cuda()
            token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_types[minibatch_index * 16: (minibatch_index + 1) * 16], batch_first=True).cuda()
            
            with torch.no_grad():
                logits = self.extraction_model(input_ids, token_type_ids)
                for index, logit in enumerate(logits):
                    input_ids = batch_ids[minibatch_index * 16 + index]
                    len_id = len(input_ids)
                    offset = batch_offset[minibatch_index * 16 + index]
                    offset_mapping = batch_mapping[minibatch_index * 16 + index]
                    output = self._recognize(self.text, logit[0][offset:len_id,offset:len_id], offset_mapping[offset:])
                    #output = [(i, self.sigmoid(j)) for (i,j) in output]
                    if output == []:
                        output.append(('[None]', 0.9, 3))
                    entities_2step.append(self.score2prob(output))
        
        prob_num = 0
        # print(entities_2step)
        for i, second_entity_list in enumerate(entities_2step):
            key1, key2 = batch_key[i]
            for spo in self.spo_list[cond]:
                #print(spo)
                for second_entity in second_entity_list:
                    if isinstance(spo[key1], tuple):
                        if spo[key1][0] == entities_1step[i // len(new_choices)][0] and spo[key2][0] == second_entity[0]:
                            # prob_num += entities_1step[i // len(new_choices)][1] * second_entity[1] * second_entity[2]
                            prob_num += second_entity[2]
                    else:
                        if spo[key1] == entities_1step[i // len(new_choices)][0] and spo[key2] == second_entity[0]:
                            # prob_num += entities_1step[i // len(new_choices)][1] * second_entity[1] * second_entity[2]
                            prob_num += second_entity[2]
        #print(prob_num)
        return prob_num, entities_1step

    def step(self, cond, action, choices):
        '''
        Action就是预测下一个待抽取的槽位,
        一个action需要返回多个下个状态（如果抽取多个候选要素，进行状态的分裂）
        return:
        Reward, []
        '''
        """ if self.lang == 'en':
            new_cond = f'{cond}; {choices[action]}:'
        elif self.lang == 'zh':
            new_cond = f'{cond}； {choices[action]}：'
        new_choices = copy.deepcopy(choices)
        del new_choices[action]
        input_ids, token_type_ids, offset, offset_mapping = self._example_generation(self.text, new_cond)
        
        with torch.no_grad():
            logits = self.extraction_model(input_ids.unsqueeze(0), token_type_ids.unsqueeze(0))
            entities_1step = self._recognize(self.text, logits[0,0][offset:,offset:], offset_mapping[offset:])
        
        entities_2step = []
        batch_ids, batch_token_types, batch_offset, batch_mapping = []
        for entity in entities_1step:
            for choice in new_choices:
                if self.lang == 'en':
                    tmp_cond = f'{new_cond}{entity[0]}; {choice}:'
                elif self.lang == 'zh':
                    tmp_cond = f'{new_cond}{entity[0]}； {choice}：'
                input_ids, token_type_ids, offset, offset_mapping = self._example_generation(self.text, tmp_cond)
                batch_ids.append(input_ids)
                batch_token_types.append(token_type_ids)
                batch_offset.append(offset)
                batch_mapping.append(offset_mapping)

        for minibatch_index in range((len(batch_ids) - 1) // 16 + 1):
            input_ids = torch.nn.utils.rnn.pad_sequence(batch_ids[minibatch_index * 16: (minibatch_index + 1) * 16], batch_first=True)
            token_type_ids = torch.nn.utils.rnn.pad_packed_sequence(batch_token_types[minibatch_index * 16: (minibatch_index + 1) * 16], batch_first=True)

            logits = self.extraction_model(input_ids, token_type_ids)
            for index, logit in enumerate(logits):
                offset = batch_offset[minibatch_index * 16 + index]
                offset_mapping = batch_mapping[minibatch_index * 16 + index]
                entities_2step.append(self._recognize(self.text, logit[0][offset:,offset:], offset_mapping[offset:]))

        reward, valid_conds = self.getReward(choices[action], entities, cond) """
        ##########################################################
        # FIXME: 这是根据信息增益定义的Reward Function
        """ slot_name = choices[action]
        prob_num, entities = self.choice_decision(cond, choices, action, step=2)
        reward = prob_num
        #print(reward)
        if self.mode == 'train':
        #if self.mode:
            for act in range(len(choices)):
                if act != action:
                    tmp_prob = self.choice_decision(cond, choices, act, step=1)
                    #print(tmp_prob)
                    #reward += prob_num - tmp_prob
                    reward -= tmp_prob

        reward = reward / self.gt_num """
        ##########################################################
        slot_name = choices[action]
        entities = self.choice_decision(cond, choices, action, step=0)
        reward = sum([entity[2] for entity in entities]) / len(entities)
        """ print('*'*50)
        print(self.text)
        print(cond)
        print(self.spo_list[cond])
        print(entities)
        print(reward) """

        entities = list(set([e[0] for e in entities]))
        valid_conds = []
        for entity in entities:
            if self.lang == 'en':
                new_cond = f'{cond}; {slot_name}:{entity}'
            elif self.lang == 'zh':
                new_cond = f'{cond}； {slot_name}：{entity}'
            if new_cond not in self.spo_list.keys():
                self.spo_list[new_cond] = []
            valid_conds.append(new_cond)
            for spo in self.spo_list[cond]:
                if self.dname == 'DuEE1.0':
                    if spo[slot_name][0] == entity:
                        self.spo_list[new_cond].append(spo)
                else:
                    if spo[slot_name] == entity:
                        self.spo_list[new_cond].append(spo)

        new_choices = copy.deepcopy(choices)
        del new_choices[action]

        if new_choices:
            done = False
        else:
            done = True
        #return
        return [(_cond , self.text, new_choices) for _cond in valid_conds], reward, done

    def _recognize(self, text, logits, offset_mapping, threshold = 0):
        '''
        根据GlobalPointer模型的输出进行抽取结果的识别，返回形式：
            [(entity, scores),...]
        '''
        scores = logits.squeeze().cpu() # Size: [seqlen, seqlen]
        entities = []
        for start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (text[offset_mapping[start][0]: offset_mapping[end][1]], scores[start,end])
            )
        if entities:
            entities.sort(key=lambda x:x[-1], reverse=True)
        
        return entities

    def getReward_2(self, entities_1step, entities_2step):
        '''
        根据两步抽取模型的结果计算信息熵，来反馈计算Reward
        '''
        
        pass

    def getReward(self, slot_name, entities, cond):
        '''
        根据抽取模型的结果进行Reward的反馈：
            𝑟(𝑠𝑡𝑎𝑡𝑒,𝑎𝑐𝑡𝑖𝑜𝑛)=𝐶𝑂𝑉(𝑟𝑒𝑠)+𝑀𝑎𝑡𝑐ℎ(𝑟𝑒𝑠)+𝐾𝑛𝑜𝑤𝑙𝑒𝑑𝑔𝑒(𝑟𝑒𝑠) \n
        目前没有候选结果的离散程度（即第一项）& 先验知识（即最后一项）\n
        𝑀𝑎𝑡𝑐ℎ(𝑟𝑒𝑠)根据可正确预测的多元组给相应的打分（均为正值）
        '''
        #ground_truth_spo_num = len(self.spo_list[cond])
        predict_truth_spo_num = 0
        valid_conds = []
        if entities == []:
            entities.append(('[None]', -10))
        #print(entities)
        #entities = list(set([e[0] for e in entities]))
        entities_mention = list(set([e[0] for e in entities]))
        # 计算每个Entity出现的概率
        logsum = sum([math.exp(e[1]) for e in entities])
        entities = [(e[0],math.exp(e[1])/logsum) for e in entities]
        entities_score = [(name, sum([i[1] for i in entities if i[0] == name])) for name in entities_mention]
        #print(entities_score)
        #print(self.spo_list[cond])
        #print(slot_name)
        #print('*'*100)

        for entity in entities_score:
            if self.lang == 'en':
                new_cond = f'{cond}; {slot_name}:{entity[0]}'
            elif self.lang == 'zh':
                new_cond = f'{cond}； {slot_name}：{entity[0]}'
            if new_cond not in self.spo_list.keys():
                self.spo_list[new_cond] = []
            valid_conds.append(new_cond)

            if self.dname == 'WebNLG':
                for spo in self.spo_list[cond]:
                    if spo[slot_name] == entity[0]:
                        self.spo_list[new_cond].append(spo)
                        predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1]
                        #predict_truth_spo.append(spo)
                        #break
            elif self.dname == 'DuEE1.0':
                if self.reward_type in ['v1', 'v3']:
                    for spo in self.spo_list[cond]:
                        if spo[slot_name][0] == entity[0]:
                            self.spo_list[new_cond].append(spo)
                            predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1]
                    """ for _spo in self.spo_list[cond]:
                        spo = _spo['arguments']
                        if slot_name in spo.keys():
                            for sn in spo[slot_name]:
                                if sn['argument'] == entity[0]:
                                    self.spo_list[new_cond].append(_spo)
                                    predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1] """
                            #break
                else:
                    for _spo in self.spo_list[cond]:
                        spo = _spo['arguments']
                        if slot_name in spo.keys():
                            max_f1 = 0
                            for sn in spo[slot_name]:
                                max_f1 = max(max_f1, text_f1(entity[0], sn['argument']))
                                if sn['argument'] == entity[0]:
                                    self.spo_list[new_cond].append(_spo)
                            predict_truth_spo_num += max_f1
                    #print(max_f1)
            elif self.dname == 'DuIE2.0':
                for spo in self.spo_list[cond]:
                    #print(spo)
                    if spo[slot_name] == entity[0]:
                        self.spo_list[new_cond].append(spo)
                        predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1]
                        #break
            elif self.dname == 'DuEE-fin':
                for spo in self.spo_list[cond]:
                    if spo[slot_name] == entity[0]:
                        self.spo_list[new_cond].append(spo)
                        predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1]
            elif self.dname == 'HacRED':
                for spo in self.spo_list[cond]:
                    if spo[slot_name] == entity[0]:
                        self.spo_list[new_cond].append(spo)
                        predict_truth_spo_num += 1 + (self.reward_type == 'v3') * entity[1]
            
        reward = predict_truth_spo_num
        #self.spo_list = predict_truth_spo # 做答案的更新
        return reward, valid_conds

    def getState(self, cond, choices):
        '''State包含：
        1. 提示cond
        2. 文本Text
        3. 选项Choice
        环境还是以文本为主，由cond + sent组成
        '''
        cond_tokens = self.tokenizer.tokenize(cond)
        text_tokens = self.tokenizer.tokenize(self.text)
        mask_index = []
        # 构造Cond
        tokens = [self.tokenizer.cls_token] + cond_tokens
        choice_mask = [0] * len(tokens)
        
        tokens.append(self.tokenizer.sep_token)
        choice_mask.extend([0])
        cond_offset = len(tokens)

        # 构造Text
        tokens.extend(text_tokens)
        choice_mask.extend([0] * len(text_tokens))
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:512])
        choice_mask = choice_mask[:512]

        token_type_ids = torch.zeros(len(input_ids)).int()
        token_type_ids[cond_offset:] = 1

        assert len(input_ids) == len(choice_mask)

        return [torch.IntTensor(input_ids), \
                token_type_ids, \
                torch.IntTensor(choice_mask), \
                mask_index
        ]

    """ def getState(self, cond, choices):
        '''State包含：
        1. 提示cond
        2. 文本Text
        3. 选项Choice
        环境还是以文本为主，由cond + sent组成
        目前将环境做成一个Multichoice的问题
        '''
        cond_tokens = self.tokenizer.tokenize(cond)
        text_tokens = self.tokenizer.tokenize(self.text)
        choice_tokens_list = [self.tokenizer.tokenize(choice) for choice in choices]
        mask_index = []
        # 构造Cond
        tokens = [self.tokenizer.cls_token] + cond_tokens
        choice_mask = [0] * len(tokens)

        for choice_tokens in choice_tokens_list:
            tokens.extend([self.tokenizer.sep_token] + choice_tokens)
            mask_index.append(len(choice_mask))
            choice_mask.extend([1] + [0] * len(choice_tokens))
        
        tokens.append(self.tokenizer.sep_token)
        choice_mask.extend([0])
        cond_offset = len(tokens)

        # 构造Text
        tokens.extend(text_tokens)
        choice_mask.extend([0] * len(text_tokens))
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:512])
        choice_mask = choice_mask[:512]

        token_type_ids = torch.zeros(len(input_ids)).int()
        token_type_ids[cond_offset:] = 1

        assert len(input_ids) == len(choice_mask)

        return [torch.IntTensor(input_ids), \
                token_type_ids, \
                torch.IntTensor(choice_mask), \
                mask_index
        ]
    """

    def return_cond(self):
        return self.spo_list

    def slot_fill_(self, slot_list, cond):
        if self.dname == 'DuEE1.0':
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        rel[slot_name] = ('[None]',-1)
        else:
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        rel[slot_name] = '[None]'

    def reset_with_input(self, text, cond, choices):
        self.spo_list = {}
        self.spo_list[cond] = {}
        self.text = text
        self.gt_num = 1e12
        return [(cond, text, choices)], 0, False

    def reset(self):
        '''
        data的形式: [
            Text, 
            predicate, 
            (s,o) list for the corresponding relation
        ]
        '''
        self.spo_list = {}
        if self.mode == 'train':
            index = self.index % self.dataset_len
            self.index += 1
            #index = random.randint(0,self.dataset_len - 1)
            #index = 24
        else:
            index = self.index
            self.index += 1

        self.data = self.dataset[index]
        cond = self.data[1]
        text = self.data[0]
        self.gt_num = len(self.data[2])
        self.spo_list[cond] = self.data[2]
        #print(self.spo_list)
        if self.dname == 'WebNLG' or self.dname == 'NYT':
            choices = ['subject', 'object']
        elif self.dname in ['HacRED','SKE']:
            choices = ['头实体','尾实体']
        elif self.dname == 'DuEE1.0':
            choices = self.schema[cond] + ['触发词']
            # slot 补全
            self.slot_fill_(choices, cond)
        elif self.dname == 'DuIE2.0':
            choices = self.schema[cond]
            self.slot_fill_(choices, cond)
        elif self.dname == 'DuEE-fin':
            choices = self.schema[cond] + ['触发词']
            self.slot_fill_(choices, cond)
        self.text = text
        return [(cond, self.text, choices)], 0, False

if __name__ == '__main__':
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-bert-wwm-ext')
    """ env = ExtractionEnv(plm='hfl/chinese-bert-wwm-ext', 
                        extraction_model_weight='weight/gp_duee.pt', 
                        tokenizer=tokenizer, 
                        data_path='data/DuEE1.0/duee_train.json', 
                        dataset='DuEE1.0',
                        lang='zh') """
    """ env = ExtractionEnv(plm='hfl/chinese-bert-wwm-ext',
                        extraction_model_weight='weight/gp_hacred.pt',
                        tokenizer=tokenizer,
                        data_path='data/HacRED/new_valid.json',
                        dataset='HacRED',
                        lang='zh') """
    env = ExtractionEnv(plm='bert-base-cased',
                        extraction_model_weight='weight/version1/gp_nyt.pt',
                        tokenizer=tokenizer,
                        data_path='data/NYT10/new_train.json',
                        dataset='NYT',
                        lang='en')

    state_list, reward, done = env.reset()
    #print(state_list, reward, done)
    #print(reward)
    while not done:
        print('*'*100)
        cond, _, choices = state_list[0]
        state_list, reward, done = env.step(cond, 0, choices)
        #print(state_list, reward, done)
        print(reward)

    #cond, _, choices = state_list[0]
    #state_list, reward, done = env.step(cond, 0, choices)
    #print(reward)
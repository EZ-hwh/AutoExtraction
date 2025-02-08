import os, sys
from .replay_buffer import Memory
sys.path.append('../')
import pt_utils
from model import ActorModel, ActorModel_grtxl
import torch
import random
import torch.nn as nn
import wandb

class MemoryBank:
    def __init__(self, buf_sz=10000):
        self.buffer = []
        self.buf_sz = buf_sz

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.buf_sz:
            self.buffer = self.buffer[self.buf_sz//2:]
        self.buffer.append((state, action, reward, next_state, done))

class DQN:
    def __init__(self, plm, epsilon, tokenizer, gamma, buf_sz, batch_sz, lr, explore_update):
        self.batch_sz = batch_sz
        self.tokenizer = tokenizer
        self.memory = Memory(batch_size=batch_sz, max_size=buf_sz, beta=0.9)
        self.epsilon = epsilon
        self.gamma = gamma
        self.ucnt = 0
        self.explore_update = explore_update
        self.policy_net = ActorModel(plm).cuda()
        self.target_net = ActorModel(plm).cuda()
        #pt_utils.lock_transformer_layers(self.policy_net.bert, 10)
        """ self.policy_net = ActorModel_grtxl(plm).cuda()
        self.target_net = ActorModel_grtxl(plm).cuda() """
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # self.opt = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.opt = torch.optim.SGD(self.policy_net.parameters(), lr=lr)

    def select_action(self, cond, text, choices):
        # TODO: 按道理来说选择Action可以弄成并行加速一下路径采样
        # Exploration
        if random.random() < self.epsilon:
            return torch.rand(len(choices)) 
            #random.choice(list(range(len(choices))))]

        # Exploit
        input_ids, token_type_ids = [], []
        for choice in choices:
            output = self.tokenizer(choice + ' ' + cond, text, return_token_type_ids=True)
            input_ids.append(torch.IntTensor(output['input_ids'][:512]))
            token_type_ids.append(torch.IntTensor(output['token_type_ids'][:512]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids
            }
            logits = self.policy_net(**inputs)
        return logits.squeeze()

    def select_next_action(self, cond, text, choices):
        if choices == []:
            return torch.tensor(0).cuda()
        # Exploit
        input_ids, token_type_ids = [], []
        for choice in choices:
            output = self.tokenizer(choice + ' ' + cond, text, return_token_type_ids=True)
            input_ids.append(torch.IntTensor(output['input_ids'][:512]))
            token_type_ids.append(torch.IntTensor(output['token_type_ids'][:512]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids
            }
            logits = self.target_net(**inputs)
        return torch.max(logits.squeeze(), dim=0)[0]

    def getBatch(self):
        points, batch, importance_ratio = self.memory.get_mini_batches()
        state, action, reward, next_state_list, done = zip(*batch)
        
        flatten_next_state = [j for i in next_state_list for j in i]
        flatten_offset = [len(i) for i in next_state_list]

        cond_, text_, choices_ = zip(*flatten_next_state)
        input_ids = []
        token_type_ids = []
        for (cond, text, choices),action in zip(state, action):
            output = self.tokenizer(choices[action] + ' ' + cond, text, return_token_type_ids=True)
            input_ids.append(torch.IntTensor(output['input_ids'][:512]))
            token_type_ids.append(torch.IntTensor(output['token_type_ids'][:512]))
        
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                    padding_value=0,
                                    batch_first=True).cuda()

        reward = torch.tensor(reward).cuda()
        done = torch.tensor(done).cuda()
            
        return input_ids, token_type_ids, \
                reward, \
                cond_, text_, choices_, \
                done, points, importance_ratio, \
                flatten_offset

    def store_transition(self, state, action, reward, next_state_list, done):
        self.memory.store_transition(state, action, reward, next_state_list, done)

    def chunk_sum(self, t, split_list):
        '''
            Input: torch.Tensor([1,2,3,4,5,6,7,8]), [2,2,4]
            Output: torch.Tensor([(1+2), (3+4), (5+6+7+8)])
        '''
        ch_sum = torch.zeros(len(split_list)).cuda()
        for i, chunk in enumerate(torch.split(t, split_list)):
            ch_sum[i] = torch.mean(chunk)
        return ch_sum

    def update(self):
        #print('='*50)
        
        self.ucnt += 1
        if self.ucnt % self.explore_update == 0 and self.epsilon > 0.02: self.epsilon *= 0.95

        input_ids, token_type_ids, \
        reward, \
        cond_, text_, choices_, \
        done, points, importance_ratio, \
        flatten_offset = self.getBatch()
        
        next_q = torch.tensor([self.select_next_action(c, t, ch) for (c,t,ch) in zip(cond_, text_, choices_)])
        ground_truth = reward + self.gamma * self.chunk_sum(next_q, flatten_offset) * (1 - done.long())
        
        out = self.policy_net(input_ids, token_type_ids).squeeze()
        td_error = torch.abs(out - ground_truth)

        loss = nn.MSELoss()(out,ground_truth)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        wandb.log({"loss": loss.cpu(), "td_error": td_error.mean().cpu()})

        self.memory.update(points, td_error.cpu().detach().numpy())
        #print(loss)

    def save_weight(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weight(self, path):
        self.policy_net.load_state_dict(torch.load(path))


if __name__ == '__main__':
    d = DQN
    a = torch.rand(10).cuda()
    b = [2,1,3,4]
    print(a)
    print(d.chunk_sum('', a,b))
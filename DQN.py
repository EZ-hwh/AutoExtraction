from model import ActorModel
import torch
import torch.nn as nn
import random
from transformers import BertTokenizerFast, RobertaTokenizerFast, set_seed
from Environment import ExtractionEnv
import argparse
import numpy as np
from tqdm import tqdm
from utils import *
from collections import Counter
from RL_utils.dqn import DQN
import math
import time
import wandb

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--plm", type=str, default='bert-base-cased', help='pretrain language model for bert')
parser.add_argument("--exploration_update", type=int, default=10000, help='how many step to update exploration ratio')
parser.add_argument("--max_step", type=int, default=20, help='max step on a single episode')
parser.add_argument("--num_episode", type=int, default=10, help='sample episode number')
parser.add_argument("--learning_rate", type=float, default=1e-5, help='Learning rate for the DQN model')
parser.add_argument("--do_train", type=bool, default=False, help='whether train')
parser.add_argument("--do_test", type=bool, default=False, help='whether test')
parser.add_argument("--weight_file", type=str, default='rl_webnlg.pt', help='path to save rl model weight')
parser.add_argument('--dname', type=str, choices=['WebNLG','DuEE1.0','DuEE-fin','DuIE2.0','HacRED','NYT','SKE'])
parser.add_argument('--buf_size', type=int, default=10000, help='the size of the memory size')
parser.add_argument('--action_strategy', type=str, default='RL', choices=['RL','Random','Sequence'])
parser.add_argument('--reward_type', type=str, choices=['v1', 'v2', 'v3'], default='v1')
parser.add_argument('--data_split', type=int, choices=[1,2], default=1)
args = parser.parse_args()

seed = args.seed
plm = args.plm
lr = args.learning_rate
target_update = 20
iters_save = 3000

if args.do_train:
    """
    Training
    """
    np.random.seed(seed)
    set_seed(seed)
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_episode,
        "batch_size": 32,
        "seed": args.seed,
    }
    wandb.init(project=f"RL_{args.dname}_{args.data_split}", name=time.asctime(), config=config, notes='使用p(s|o)p(o) - p(s)进行Reward的设计')
    tokenizer = BertTokenizerFast.from_pretrained(plm)
    if args.dname == 'WebNLG':
        env = ExtractionEnv(plm=plm, 
                        extraction_model_weight=f'weight/webnlg/ext_{3 - args.data_split}.pt', 
                        tokenizer=tokenizer, 
                        data_path='data/WebNLG/train_data.json', 
                        dataset='WebNLG',
                        reward_type=args.reward_type,
                        lang='en',
                        data_split=args.data_split)
    elif args.dname == 'DuEE1.0':
        env = ExtractionEnv(plm=plm, 
                        extraction_model_weight=f'weight/duee/ext_{3 - args.data_split}.pt', 
                        tokenizer=tokenizer,
                        data_path='data/DuEE1.0/duee_train.json', 
                        dataset='DuEE1.0',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'DuIE2.0':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/duie/ext_{3 - args.data_split}.pt',
                        tokenizer=tokenizer,
                        data_path='data/DuIE2.0/duie_train.json',
                        dataset='DuIE2.0',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'DuEE-fin':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/duee_fin/ext_{3 - args.data_split}.pt',
                        tokenizer=tokenizer,
                        data_path='data/DuEE-fin/duee_fin_train_new.json',
                        dataset='DuEE-fin',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'HacRED':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/hacred/ext_{3 - args.data_split}.pt',
                        tokenizer=tokenizer,
                        data_path='data/HacRED/new_train.json',
                        dataset='HacRED',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'NYT':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/nyt/ext_{3 - args.data_split}.pt',
                        tokenizer=tokenizer,
                        data_path='data/NYT10/new_train.json',
                        dataset='NYT',
                        reward_type=args.reward_type,
                        lang='en',
                        data_split=args.data_split)
    elif args.dname == 'SKE':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/ske/ext_{3 - args.data_split}.pt',
                        tokenizer=tokenizer,
                        data_path='data/ske2019/new_train.json',
                        dataset='SKE',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)

    tot_step = args.num_episode * env.dataset_len
    n = math.log(0.05 / 0.9) / math.log(0.95)
    explore_update = int(tot_step // n)
    print('Explore update:', explore_update)
    agent = DQN(plm=plm,epsilon=0.9, tokenizer=tokenizer, gamma=0.5,buf_sz=args.buf_size,batch_sz=32, lr=lr, explore_update=explore_update)
    
    print('Begin RL training!')

    rewards = []
    moving_average_rewards = [] 
    ep_steps = []
    for i_episode in tqdm(range(tot_step), desc='Training RL agent'): 
        state_list, _, _ = env.reset()
        ep_reward = 0
        for i_step in range(args.max_step): 
            new_state_list = []
            for state in state_list:
                cond, text, choices = state
                action = agent.select_action(cond, text, choices)
                action = torch.argmax(action)
                next_state_list, reward, done = env.step(cond, action, choices)
                ep_reward += reward / len(state_list)

                agent.store_transition(state, action, reward, next_state_list, done)
                """ for next_state in next_state_list:
                    agent.store_transition(state, action[1], reward, next_state, done) """
                new_state_list.extend(next_state_list)
                wandb.log({'reward': reward})
            state_list = new_state_list
            if done:
                break
        agent.update()
        wandb.log({'ep_reward': ep_reward})
        
        if i_episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if i_episode % iters_save == 0 and i_episode != 0:
            agent.save_weight(f'weight/{args.weight_file}/{args.reward_type}_rl_{args.data_split}.pt')

        ep_steps.append(i_step)
        rewards.append(ep_reward)
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1] + 0.1*ep_reward)

if args.do_test:
    tokenizer = BertTokenizerFast.from_pretrained(plm)

    if args.dname == 'WebNLG':
        env = ExtractionEnv(plm=plm, 
                        extraction_model_weight='weight/version1/gp_webnlg.pt', 
                        tokenizer=tokenizer, 
                        data_path='data/WebNLG/test_filter.json',
                        dataset='WebNLG',
                        mode='test',
                        lang='en')
    elif args.dname == 'DuEE1.0':
        env = ExtractionEnv(plm=plm, 
                        extraction_model_weight='weight/version1/gp_duee.pt', 
                        tokenizer=tokenizer, 
                        data_path='data/DuEE1.0/duee_dev_filter.json', 
                        dataset='DuEE1.0',
                        mode='test',
                        lang='zh')
    elif args.dname == 'DuIE2.0':
        env = ExtractionEnv(plm=plm, 
                        extraction_model_weight='weight/version1/gp_duie.pt',
                        tokenizer=tokenizer, 
                        data_path='data/DuIE2.0/duie_dev.json', 
                        dataset='DuIE2.0',
                        mode='test',
                        lang='zh')
    elif args.dname == 'DuEE-fin':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight='weight/version1/gp_duee_fin.pt',
                        tokenizer=tokenizer,
                        data_path='data/DuEE-fin/duee_fin_dev_new_filter.json',
                        dataset='DuEE-fin',
                        mode='test',
                        lang='zh')
    elif args.dname == 'HacRED':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight='weight/version1/gp_hacred.pt',
                        tokenizer=tokenizer,
                        data_path='data/HacRED/new_test_filter.json',
                        dataset='HacRED',
                        mode='test',
                        lang='zh')
    elif args.dname == 'NYT':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight='weight/version1/gp_nyt.pt',
                        tokenizer=tokenizer,
                        data_path='data/NYT/new_test.json',
                        dataset='NYT',
                        mode='test',
                        lang='en')
    elif args.dname == 'SKE':
        env = ExtractionEnv(plm=plm,
                        extraction_model_weight=f'weight/version1/gp_ske.pt',
                        tokenizer=tokenizer,
                        data_path='data/ske2019/new_test.json',
                        dataset='SKE',
                        mode='test',
                        lang='zh')

    if args.action_strategy == 'RL':
        agent1 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=lr, explore_update = 1e10)
        agent1.load_weight(f'weight/{args.weight_file}/v2_rl_1.pt')
        agent2 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=lr, explore_update = 1e10)
        agent2.load_weight(f'weight/{args.weight_file}/v2_rl_2.pt')

    print('Begin RL testing!')
    tp, tn, fp = 0, 0, 0
    pre_role, gt_role, tot_f1 = 0, 0, 0
    tot = 0

    for i_episode in tqdm(range(env.dataset_len),desc='Extract through RL agent'): 
        state_list, _, _ = env.reset()
        # Restrict slot_num
        slot_list = state_list[0][2]
        slot_num = len(slot_list)
        #print(slot_num)
        if slot_num <= 0:
            continue
        ep_reward = 0
        for i_step in range(args.max_step): 
            new_state_list = []
            for state in state_list:
                cond, text, choices = state
                # Auto extraction
                if args.action_strategy == 'RL':
                    action1 = agent1.select_action(cond, text, choices)
                    #print(action1)
                    #action = action1[1]
                    action2 = agent2.select_action(cond, text, choices)
                    action = torch.argmax(action1 + action2)
                    #print(action1+action2)
                    """ if action1[0] > action2[0]:
                        action = action1[1]
                    else:
                        action = action2[1] """
                    #print(action, choices, cond)
                # Random
                elif args.action_strategy == 'Random':
                    action = random.randint(0, len(choices) - 1)
                # Seq
                elif args.action_strategy == 'Sequence':
                    action = 0
                next_state_list, reward, done = env.step(cond, action, choices) 
                new_state_list.extend(next_state_list)
            state_list = new_state_list
            if done:
                break
        #print(env.return_cond().keys())
        predict_list = env.return_cond()
        #print(predict_list)
        if args.dname == 'WebNLG':
            for k in predict_list.keys():
                c = Counter(k)
                if c[';'] == 0:
                    #print(predict_list[k])
                    tot += len(predict_list[k])
                if (c[';'] == slot_num) and ('[None]' not in k):
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
        elif args.dname in ['NYT']:
            for k in predict_list.keys():
                c = Counter(k)
                if c[';'] == 0:
                    tot += len(predict_list[k])
                if (c[';'] == slot_num) and ('[None]' not in k):
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
        elif args.dname == 'DuEE1.0':
            #slot_num = 0
            ground_truth = []
            for k in predict_list.keys():
                c = Counter(k)
                if c['；'] == 0:
                    tot += len(predict_list[k])
                elif c['；'] == slot_num:
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
            ground_truth = {}
            pre_list = []
            for k in predict_list.keys():
                pre = {}
                c = Counter(k)
                if c['；'] == 0:
                    gt = predict_list[k]
                if c['；'] == slot_num:
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

            f1_score = np.zeros((len(pre_list), len(gt)))
            for i, pre in enumerate(pre_list):
                match = False
                for j, g_tuple in enumerate(gt):
                    f1_score[i,j] = calc_f1_new(pre, g_tuple)
                    if pre == g_tuple:
                        match = True
                """ if match:
                    tp += 1
                else:
                    tn += 1 """
            #tot += len(gt)
            pre_role += sum(f1_score.max(1))
            gt_role += sum(f1_score.max(0))

        elif args.dname == 'DuIE2.0':
            #slot_num = 0
            for k in predict_list.keys():
                c = Counter(k)
                if c['；'] == 0:
                    tot += len(predict_list[k])
                    #slot_num = len(predict_list[k][0].keys())
                elif c['；'] == slot_num:
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
        elif args.dname == 'DuEE-fin':
            #print('*'* 50)
            #print(predict_list)
            for k in predict_list.keys():
                c = Counter(k)
                if c['；'] == 0:
                    tot += len(predict_list[k])
                if c['；'] == slot_num:
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
            ground_truth = {}
            pre_list = []
            for k in predict_list.keys():
                pre = {}
                c = Counter(k)
                if c['；'] == 0:
                    gt = predict_list[k]
                if c['；'] == slot_num:
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

            f1_score = np.zeros((len(pre_list), len(gt)))
            for i, pre in enumerate(pre_list):
                match = False
                for j, g_tuple in enumerate(gt):
                    f1_score[i,j] = calc_f1_new(pre, g_tuple)
                    if pre == g_tuple:
                        match = True
                """ if match:
                    tp += 1
                else:
                    tn += 1 """
            #tot += len(gt)
            pre_role += sum(f1_score.max(1))
            gt_role += sum(f1_score.max(0))
        elif args.dname in ['HacRED','SKE']:
            for k in predict_list.keys():
                c = Counter(k)
                if c['；'] == 0:
                    tot += len(predict_list[k])
                if (c['；'] == slot_num) and ('[None]' not in k):
                    if predict_list[k] != []:
                        tp += len(predict_list[k])
                    else:
                        tn += 1
        """ if i_episode > 5:
            break """
    prec = tp / (tp + tn)
    reca = tp / tot
    f1 = 2 * prec * reca / (prec + reca)

    print(f'Prec:   {tp}/{tp+tn} = {prec:.4f}')
    print(f'Reca:   {tp}/{tot} = {reca:.4f}')
    print(f'F1:     {f1:.4f}')

    if args.dname.startswith('DuEE'):
        prec = pre_role / (tp + tn)
        reca = gt_role / tot
        f1 = 2 * prec * reca / (prec + reca)
        print(f'Char Prec:   {prec:.4f}')
        print(f'Char Reca:   {reca:.4f}')
        print(f'Char F1:     {f1:.4f}')

import os, sys, json
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch
from transformers import BertTokenizerFast, set_seed

# Self package
sys.path.append('../')
from data_utils import WebNLGDataset, DuEEDataset, DuIEDataset, DuEE_finDataset, HacREDDataset
from model import GlobalPointerModel, global_pointer_crossentropy, global_pointer_f1_score
import pt_utils

# Global params
LR = 2e-5
batch_size = 32
epochs = 10
set_seed(42)

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", help='training extraction model', type=bool, default=False)
parser.add_argument("--do_test", help='predict the knowledge tuple in texts.', type=bool, default=False)
parser.add_argument("--plm", help='backbone pretrained language model', default='bert-base-cased')
parser.add_argument("--weight_file", help='path to save the model weight.', default='gp_webnlg.pt', type=str)
parser.add_argument("--dname", help='dataset for train or test', choices=['WebNLG','DuEE1.0','DuEE-fin','DuIE2.0', 'HacRED'], type=str)
parser.add_argument("--load_old", help='whether load old weight for training.', default=False, type=bool)
parser.add_argument("--data_split", help='choose the split of the train data.', default=1, choices=[1,2], type=int)
args = parser.parse_args()


class RCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.fc = nn.Linear(768, len(rel_map))
    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z[:,0,:])
        out = torch.sigmoid(out)
        return out

if args.do_train:
    weight_file = f'weight/{args.weight_file}'
    tokenizer = BertTokenizerFast.from_pretrained(args.plm)
    if args.dname == 'WebNLG':
        dataset = WebNLGDataset('data/WebNLG/train_data.json', tokenizer, data_split=args.data_split)
    elif args.dname == 'DuEE1.0':
        dataset = DuEEDataset('data/DuEE1.0/duee_train.json', tokenizer, data_split=args.data_split)
    elif args.dname == 'DuEE-fin':
        dataset = DuEE_finDataset('data/DuEE-fin/duee_fin_train_new.json', tokenizer, data_split=args.data_split)
    elif args.dname == 'DuIE2.0':
        dataset = DuIEDataset('data/DuIE2.0/duie_train.json', tokenizer, data_split=args.data_split)
    elif args.dname == 'HacRED':
        dataset = HacREDDataset('data/HacRED/new_train.json', tokenizer, data_split=args.data_split)
    elif args.dname == 'NYT':
        dataset = NYTDataset

    train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn_cuda)
    total_steps = len(train_dl) * epochs
    model = GlobalPointerModel(args.plm).cuda()
    if args.load_old:
        if os.path.exists(weight_file):
            model.load_state_dict(torch.load(weight_file))
        else:
            print('The weight file is not exists!')
    #model = TripleGeneration(args.plm).cuda()
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, LR, total_steps)

    pt_utils.lock_transformer_layers(model.bert, 6)

    loss_fct = global_pointer_crossentropy
    def train_func(model, batch):
        input_ids, token_type_ids, labels = batch
        
        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }
        logits = model(**inputs)
        loss = loss_fct(labels, logits)
        f1 = global_pointer_f1_score(labels, logits)
        return {'loss': loss, 'f1': f1}

    pt_utils.train_model(model, optimizer, train_dl, epochs, train_func, None,
                    scheduler=scheduler, save_file=weight_file)


if args.do_test:
    tokenizer = BertTokenizerFast.from_pretrained(args.plm)
    dataset = WebNLGDataset(tokenizer=tokenizer, data_path='dataset/WebNLG/train_data.json')
    model = GlobalPointerModel(args.plm).cuda()
    model.load_state_dict(torch.load(weight_file))

    input_ids, token_type_ids, labels = dataset[0]
    text = "Peter Stöger is manager of 1 . FC Köln which has 50000 members and participated in the 2014 season ."
    with torch.no_grad():
        inputs = {
            'input_ids': input_ids.unsqueeze(0).cuda(),
            'token_type_ids': token_type_ids.unsqueeze(0).cuda()
        }
        logits = model(**inputs)
        
        dataset._recognize(text, logits[0][0], None)


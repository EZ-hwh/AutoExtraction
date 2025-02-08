import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam
from tqdm import tqdm

class DictDataset(torch.utils.data.Dataset):
	def __init__(self, inp, labels, device='cuda', keys=None):
		if keys is None: keys = set(inp.keys())
		self.x = {k:v.to(device) for k,v in inp.items() if k in keys}
		self.labels = labels.to(device)
	def __getitem__(self, idx):
		item = {key:val[idx] for key, val in self.x.items()}
		item['labels'] = self.labels[idx]
		return item
	def __len__(self):
		return len(self.labels)

def train_pt_model(model, train_dl, criterion, optimizer, epochs=3, test_func=None, scheduler=None, data_func=None):
	if data_func is None:
		def data_func1(ditem):
			if type(ditem) is type({}):
				return {k:v.cuda() for k, v in ditem.items() if k != 'labels'}, ditem['labels'].cuda()
			if type(ditem) is type(tuple()) and len(ditem) > 2:
				return [x.cuda() for x in ditem[:-1]], ditem[-1].cuda() 
			return ditem
		data_func = data_func1
	for epoch in range(epochs):
		model.train()
		print(f'\nEpoch {epoch+1} / {epochs}:')
		pbar = tqdm(train_dl)
		iters, accloss = 0, 0
		for ditem in pbar:
			item, label = data_func(ditem)
			item, label = item.cuda(), label.cuda()
			optimizer.zero_grad()
			out = model(item)
			loss = criterion(out, label)
			iters += 1; accloss += loss
			loss.backward()
			optimizer.step()
			if scheduler: scheduler.step()
			pbar.set_postfix({'loss': f'{accloss/iters:.6f}'})
		pbar.close()
		if test_func:
			model.eval()
			test_func()

def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
				scheduler=None, save_file=None, accelerator=None):
	for epoch in range(epochs):
		model.train()
		print(f'\nEpoch {epoch+1} / {epochs}:')
		if accelerator:
			pbar = tqdm(train_dl, disable=not accelerator.is_local_main_process)
		else: 
			pbar = tqdm(train_dl)
		metricsums = {}
		iters, accloss = 0, 0
		for ditem in pbar:
			metrics = {}
			loss = train_func(model, ditem)
			if type(loss) is type({}):
				metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
				loss = loss['loss']
			iters += 1; accloss += loss
			optimizer.zero_grad()
			if accelerator: 
				accelerator.backward(loss)
			else: 
				loss.backward()
			optimizer.step()
			if scheduler:
				if accelerator is None or not accelerator.optimizer_step_was_skipped:
					scheduler.step()
			for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
			infos = {'loss': f'{accloss/iters:.6f}'}
			for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
			pbar.set_postfix(infos)
		pbar.close()
		if save_file:
			if accelerator:
				accelerator.wait_for_everyone()
				unwrapped_model = accelerator.unwrap_model(model)
				accelerator.save(unwrapped_model.state_dict(), save_file)
			else:
				torch.save(model.state_dict(), save_file)
		if test_func:
			if accelerator is None or accelerator.is_local_main_process: 
				model.eval()
				test_func()

def lock_transformer_layers(bert, num_locks):
	import ljqpy
	num = 0
	for name, param in bert.named_parameters():
		if 'embeddings.' in name: ll = -1
		else: 
			ll = int('0'+ljqpy.RM('encoder.layer.([0-9]+)\\.', name))
		if ll < num_locks:
			#print(f'locking {name}')
			num += 1
			param.requires_grad = False
	print(f'Locked {num} parameters ...')

def get_bert_adamw(model, lr=1e-4):
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
	]
	return AdamW(optimizer_grouped_parameters, lr=lr)
	#return Adam(optimizer_grouped_parameters, lr=lr)

def get_bert_optim_and_sche(model, lr, total_steps):
	optimizer = get_bert_adamw(model, lr=lr)
	scheduler = get_linear_schedule_with_warmup(optimizer, total_steps//10, total_steps)
	return optimizer, scheduler

'''
def train_model(model, optimizer, train_dl, valid_dl = None, epochs=3, train_func=None, valid_func=None, 
				scheduler=None, save_file=None, accelerator=None):

	best_acc = 100 
	best_epoch = 0 
	for epoch in range(epochs):
		model.train()
		print(f'\nEpoch {epoch+1} / {epochs}:')
		if accelerator:
			pbar = tqdm(train_dl, disable=not accelerator.is_local_main_process)
		else: 
			pbar = tqdm(train_dl)
		metricsums = {}
		iters, accloss = 0, 0
		for ditem in pbar:
			metrics = {}
			loss = train_func(model, ditem)
			if type(loss) is type({}):
				metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
				loss = loss['loss']
			iters += 1; accloss += loss
			optimizer.zero_grad()
			if accelerator: 
				accelerator.backward(loss)
			else: 
				loss.backward()
			optimizer.step()
			if scheduler:
				if accelerator is None or not accelerator.optimizer_step_was_skipped:
					scheduler.step()
			for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
			infos = {'loss': f'{accloss/iters:.6f}'}
			for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
			pbar.set_postfix(infos)
		pbar.close()
		if valid_func:
			if accelerator:
				pbar = tqdm(valid_dl, disable=not accelerator.is_local_main_process)
			else: 
				pbar = tqdm(valid_dl)
			if accelerator is None or accelerator.is_local_main_process: 
				metricsums = {}
				iters, accloss = 0, 0
				for ditem in pbar:
					model.eval()
					loss = valid_func(model, ditem)
					if type(loss) is type({}):
						metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
						loss = loss['loss']
					iters += 1; accloss += loss
					for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
					infos = {'loss': f'{accloss/iters:.6f}'}
					for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
					pbar.set_postfix(infos)
			pbar.close()

			if accloss/len(pbar) < best_acc:
				best_acc = accloss/len(pbar)
				best_epoch = epoch
				if save_file:
					if accelerator:
						accelerator.wait_for_everyone()
						unwrapped_model = accelerator.unwrap_model(model)
						accelerator.save(unwrapped_model.state_dict(), save_file)
					else:
						torch.save(model.state_dict(), save_file)
			print(f'Best loss:{best_acc} in Epoch {best_epoch}')

	print(f'Best loss:{best_acc} in Epoch {best_epoch}')
'''
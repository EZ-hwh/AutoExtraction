import json
import random

filename = 'data/NYT11-HRL/new_train.json'
with open(filename, 'r') as f:
    #lines = json.loads(f.read())
    lines = []
    for line in f.readlines():
        lines.append(line)
    data_len = len(lines)
    split_index = random.sample(list(range(data_len)), data_len // 2)

print(lines[:5])
f1 = open(filename[:-5]+'1.json', 'w', encoding='utf-8')
f2 = open(filename[:-5]+'2.json', 'w', encoding='utf-8')

for index, line in enumerate(lines):
    if index in split_index:
        f1.write(line)
    else:
        f2.write(line)
f1.close()
f2.close()

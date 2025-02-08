import json
datas = []
with open("dataset/DuEE-fin/duee_fin_train.json", encoding='utf-8') as f:
    for line in f.readlines():
        datas.append(json.loads(line))

for index, data in enumerate(datas):
    print('*'* 100)
    print(index)
    texts = data['text'].split('\n')
    if 'event_list' not in data.keys():
        continue
    for event in data['event_list']:
        event_arguments = [a['argument'] for a in event['arguments']] + [event['trigger']]
        b = False
        for text in texts:
            tmp = True
            for event_a in event_arguments:
                if event_a in text:
                    continue
                tmp = False
                break
            if tmp:
                b = True
                break
        if b == False:
            print(data['text']) 
            break
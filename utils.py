def cacl_f1(predict_dict, gt_dict):
    tot_f1 = 0
    for keys in predict_dict.keys():
        if predict_dict[keys] == [] and gt_dict[keys] == []:
            tot_f1 += 0
        elif predict_dict[keys] != [] and gt_dict[keys] != []:
            max_f1 = 0
            for a in predict_dict[keys]:
                for b in gt_dict[keys]:
                    max_f1 = max(max_f1, text_f1(a, b))
            tot_f1 += max_f1
    return tot_f1

def calc_f1_new(predict_dict, gt_dict):
    tot_f1 = 0
    for keys in predict_dict.keys():
        """ if predict_dict[keys] == '[None]' and gt_dict[keys] == '[None]':
            tot_f1 += 0 
        elif predict_dict[keys] and gt_dict[keys]: """
        tot_f1 += text_f1(predict_dict[keys], gt_dict[keys])
    return tot_f1 / len(predict_dict.keys())


def text_f1(texta, textb):
    #print(texta, textb)
    if isinstance(textb, tuple):
        textb = textb[0]
    a = [i for i in list(texta.lower()) if i != ' ']
    b = [i for i in list(textb.lower()) if i != ' ']
    tp, fp, tn = 0,0,0
    for aitem in a:
        if aitem in b:
            tp += 1
    prec = tp / (len(a) + 1e-6) 
    reca = tp / (len(b) + 1e-6)
    return 2 * prec * reca / (prec + reca + 1e-6)

if __name__ == '__main__':
    print(text_f1('Hongkou District, Shanghai', 'Yangpu District, Shanghai'))
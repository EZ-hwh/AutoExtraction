import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math
from transformers import T5Tokenizer, T5Model
from RL_utils.gtrxl import GTrXL

class ActorModel(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.output_layer = nn.Linear(768, 1)
    
    def forward(self, input_ids, token_type_ids):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = self.output_layer(last_hidden_state[:,0,:])
        return logits


""" class ActorModel(nn.Module):
    '''
    做成一个Multichoice模型，返回choice的打分
    '''
    def __init__(self, plm_name, **kwargs):
        super(ActorModel, self).__init__(**kwargs)
        self.bert = BertModel.from_pretrained(plm_name)
        self.output_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, choice_mask):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        logits = self.output_layer(last_hidden_state).squeeze(-1)
        logits = logits.masked_fill_((1 - choice_mask).bool(), -1e12)
        return logits """

class ActorModel_grtxl(nn.Module):
    def __init__(self, plm_name, **kwargs):
        super(ActorModel_grtxl, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.gtrxl = GTrXL(
            input_dim = 768,
            embedding_dim = 768
            )
        self.output_layer = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, choice_mask):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        gtrxl_output = self.gtrxl(last_hidden_state)

        logits = self.output_layer(gtrxl_output['logit']).squeeze(-1)
        logits = logits.masked_fill_((1 - choice_mask).bool(), -1e12)
        return logits

class TripleGeneration(nn.Module):
    def __init__(self, plm, **kwargs):
        super(TripleGeneration, self).__init__(**kwargs)
        self.generation_model = T5Model.from_pretrained(plm)

    def forward(self, input_ids):
        return self.generation_model(input_ids)

class ClassificationModel(nn.Module):
    def __init__(self, plm_name, output_dim, **kwargs):
        super(ClassificationModel, self).__init__(**kwargs)
        self.bert = AutoModel.from_pretrained(plm_name)
        self.output_layer = self.linear(768, output_dim)

    def forward(self, input_ids, token_type_ids):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        return self.output_layer(last_hidden_state[:,0])

class GlobalPointerModel(nn.Module):
    def __init__(self, plm_name):
        super(GlobalPointerModel, self).__init__()
        self.bert = AutoModel.from_pretrained(plm_name)
        self.gp = GlobalPointer(1,64)

    def forward(self, input_ids, token_type_ids):
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
        return self.gp(last_hidden_state)

class RCModel(nn.Module):
    def __init__(self, plm_name, num_rel):
        super().__init__()
        self.bert = AutoModel.from_pretrained(plm_name)
        self.fc = nn.Linear(768, num_rel)
    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z[:,0,:])
        out = torch.sigmoid(out)
        return out

class DoublePointerModel(nn.Module):
    def __init__(self, plm_name):
        super(DoublePointerModel, self).__init__()
        self.bert = AutoModel.from_pretrained(plm_name)
        self.output_layer = nn.Linear(768, 2)

        self.emb = nn.Embedding(6, 768)

    def forward(self, input_ids, segment_ids):
        #print(input_ids, segment_ids)
        last_hidden_state = self.bert(input_ids=input_ids, token_type_ids=segment_ids).last_hidden_state
        #print(last_hidden_state)
        logits = self.output_layer(last_hidden_state)
        start_logits , end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return F.sigmoid(start_logits), F.sigmoid(end_logits)
        #return self.gp(last_hidden_state)

def sequence_masking(x, mask, value=0.0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        if mask.dtype != x.dtype:
            mask = mask.type_as(x)
        if value == '-inf':
            value == -1e12
        elif value == 'inf':
            value == 1e12
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = x.dim() + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = mask.unsqueeze(1)
        for _ in range(x.dim() - mask.dim()):
            mask = mask.unsqueeze(mask.dim())
        return x * mask + value * (1 - mask)

class SinusoidalPositionEmbedding(nn.Module):
    """ Bert4keras:定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False, **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            seq_len = inputs.size()[1]
            inputs, positions = inputs
            if 'float' not in torch.dtype(position_ids):
                position_ids = position_ids.type(torch.FloatTensor)
            else:
                input_shape = inputs.size()
                batch_size, seq_len = input_shape[0], input_shape[1]
                position_ids = torch.arange(0, seq_len, dtype=torch.FloatType)[None]
        indices = torch.arange(0, self.output_dim // 2, dype=torch.FloatType)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = torch.tile(embeddings, [batch_size, 1, 1])
            return torch.concat([inputs, embeddings])

class GlobalPointer(nn.Module):
    def __init__(self, heads, head_size, RoPE=True, use_bias=True, tril_mask=True, kernel_initializer='lecun_normal'):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        #self.RoPE = RoPE
        self.use_bias = use_bias
        #self.tril_mask = tril_mask
        # FIXME: 线性层缺少一个lecun-normal初始化
        self.dense = nn.Linear(768, self.head_size * self.heads * 2, bias=use_bias)
        self.pe = PositionalEncoding(self.head_size)
        #self.kernel_initializer = init

    def forward(self, inputs, mask=None):
        """ inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.heads, dim=-1)
        inputs = torch.stack(inputs, dim=-2)
        qw, kw = inputs[..., : self.head_size], inputs[..., self.head_size:] """

        # Input变换
        inputs = self.dense(inputs)
        inputs_size = inputs.size() # b x l x (heads * headsz * 2)
        gpin = inputs.view(inputs_size[0], inputs_size[1], self.heads, self.head_size, 2)
        qw, kw = gpin[...,0], gpin[...,1]

        pos = self.pe(qw)
        cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, -1)
        sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, -1)

        qw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(qw.size())
        kw2 = torch.cat([-kw[..., 1::2, None], kw[..., ::2, None]], -1).view(kw.size())
        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        #mask = tf.matrix_band_part(K.ones_like(logits), 0, -1)
        mask = torch.triu(torch.ones_like(logits),diagonal=0)
        #mask = torch.linalg.band_part(torch.ones_like(logits), 0, -1)
        logits = logits - (1 - mask) * 1e12
        # scale返回
        return logits / (self.head_size**0.5)

        return logits / (self.head_size**0.5)

class PositionalEncoding(nn.Module):
    # [bst, seq, fea]
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064 。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return neg_loss + pos_loss

def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = torch.prod(torch.tensor(y_pred.size()[:2]))
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """

    y_pred = torch.greater(y_pred, 0.5)
    y_pred = y_pred.type(torch.FloatTensor).cuda()
    return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

def collate_fn_cuda(batch):
    bz = len(batch)
    maxlen = max([len(item[0]) for item in batch])
    #print(maxlen)
    batch_input_ids = []
    batch_segments = []
    batch_labels = torch.zeros((bz, 1, maxlen, maxlen))

    for index, (input_ids, segments, labels, _) in enumerate(batch):
        batch_input_ids.append(input_ids)
        batch_segments.append(segments)
        seqlen = len(input_ids)
        #print(seqlen)
        batch_labels[index][0][:seqlen,:seqlen] = labels
        #batch_labels.append(labels)

    batch_input_ids = nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
    batch_segments = nn.utils.rnn.pad_sequence(batch_segments, batch_first=True)

    return [
        batch_input_ids.cuda(), \
        batch_segments.cuda(), \
        batch_labels.cuda()
    ]
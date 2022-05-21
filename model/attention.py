import enum
import math
import copy
import json
import torch
import torch.nn as nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
import math
import numpy as np


def orignal(x):
    return x


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        print(scores.size())
        print(mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    # In our case, query in shape (batch_size,max_len,e_dim)
    # Key in shape (batch_size,max_target, e_dim)
    # Use More aggresive stargy to caluate possible

    # p_attn_tmp = torch.exp(torch.softmax(scores, dim = -1))
    # p_attn = torch.softmax(p_attn_tmp*p_attn_tmp,dim = -1)

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def attentionWithTypeScores(query, key, value, type_scores, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Use More aggresive stargy to caluate possible

    # p_attn_tmp = torch.exp(torch.softmax(scores, dim = -1))
    # p_attn = torch.softmax(p_attn_tmp*p_attn_tmp,dim = -1)

    p_attn = torch.softmax(scores, dim=-1)
    p_attn *= type_scores
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class BasicAttention(nn.Module):
    def __init__(self,
                 q_embd_size,
                 k_embd_size,
                 v_embd_size,
                 q_k_hidden_size=None,
                 output_hidden_size=None,
                 num_heads=1,  # for multi-head attention
                 score_func='scaled_dot',
                 drop_rate=0.,
                 is_q=False,  # let q_embd to be q or not,default not
                 is_k=False,
                 is_v=False,
                 bias=True
                 ):
        '''

        :param q_embd_size:
        :param k_embd_size:
        :param v_embd_size:
        :param q_k_hidden_size:
        :param output_hidden_size:
        :param num_heads: for multi-head attention
        :param score_func:
        :param is_q: let q_embd to be q or not,default not
        :param is_k: let k_embd to be k or not,default not
        :param is_v: let v_embd to be v or not,default not
        :param bias: bias of linear
        '''
        super(BasicAttention, self).__init__()
        if not q_k_hidden_size:
            q_k_hidden_size = q_embd_size
        if not output_hidden_size:
            output_hidden_size = v_embd_size
        assert q_k_hidden_size % num_heads == 0
        self.head_dim = q_k_hidden_size // num_heads
        assert self.head_dim * num_heads == q_k_hidden_size, "q_k_hidden_size must be divisible by num_heads"
        assert output_hidden_size % num_heads == 0, "output_hidden_size must be divisible by num_heads"
        if is_q:
            self.q_w = orignal
            assert q_embd_size == k_embd_size
        else:
            self.q_w = nn.Linear(q_embd_size, q_k_hidden_size, bias=bias)
        self.is_q = is_q
        self.q_embd_size = q_embd_size
        if is_k:
            self.k_w = orignal
            assert k_embd_size == q_k_hidden_size
        else:
            self.k_w = nn.Linear(k_embd_size, q_k_hidden_size, bias=bias)
        if is_v:
            self.v_w = orignal
            assert v_embd_size == output_hidden_size
        else:
            self.v_w = nn.Linear(v_embd_size, output_hidden_size, bias=bias)
        self.q_k_hidden_size = q_k_hidden_size
        self.output_hidden_size = output_hidden_size
        self.num_heads = num_heads
        self.score_func = score_func
        self.drop_rate = drop_rate

    def forward(self, q_embd, k_embd, v_embd, mask=None):
        '''
        batch-first is needed
        :param q_embd: [?,q_len,q_embd_size] or [?,q_embd_size]
        :param k_embd: [?,k_len,k_embd_size] or [?,k_embd_size]
        :param v_embd: [?,v_len,v_embd_size] or [?,v_embd_size]
        :return: [?,q_len,output_hidden_size*num_heads]
        '''
        if len(q_embd.shape) == 2:
            q_embd = torch.unsqueeze(q_embd, 1)
        if len(k_embd.shape) == 2:
            k_embd = torch.unsqueeze(k_embd, 0)
        if len(v_embd.shape) == 2:
            v_embd = torch.unsqueeze(v_embd, 0)
        batch_size = q_embd.shape[0]
        q_len = q_embd.shape[1]
        k_len = k_embd.shape[1]
        v_len = v_embd.shape[1]
        #     make sure k_len==v_len
        assert k_len == v_len

        # get q,k,v
        #         if self.is_q:
        #             q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.q_embd_size // self.num_heads)
        #             q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.q_embd_size // self.num_heads)
        #         else:
        q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.head_dim)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.head_dim)
        k = self.k_w(k_embd).view(batch_size, k_len, self.num_heads, self.head_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.head_dim)
        v = self.v_w(v_embd).view(batch_size, v_len, self.num_heads, self.output_hidden_size // self.num_heads)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.output_hidden_size // self.num_heads)

        # get score
        if isinstance(self.score_func, str):
            if self.score_func == "dot":
                score = torch.bmm(q, k.permute(0, 2, 1))

            elif self.score_func == "scaled_dot":
                temp = torch.bmm(q, k.permute(0, 2, 1))
                score = torch.div(temp, math.sqrt(self.q_k_hidden_size))

            else:
                raise RuntimeError('invalid score function')
        elif callable(self.score_func):
            try:
                score = self.score_func(q, k)
            except Exception as e:
                print("Exception :", e)
        if mask is not None:
            mask = mask.bool().unsqueeze(1)
            score = score.masked_fill(~mask, -np.inf)
        score = nn.functional.softmax(score, dim=-1)
        score = nn.functional.dropout(score, p=self.drop_rate, training=self.training)

        # get output
        output = torch.bmm(score, v)
        heads = torch.split(output, batch_size)
        output = torch.cat(heads, -1)

        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.hidden_size = d_model
        self.d_k = self.hidden_size // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.hidden_size) for _ in range(3)])
        self.output = nn.Linear(self.hidden_size, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_attention_heads = h
        self.attention_head_size = self.d_k
        # for linears in self.linears:
        #     torch.nn.init.xavier_uniform(linears.weight)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.output(x)


class MultiHeadedAttentionWithTypeScores(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithTypeScores, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.hidden_size = d_model
        self.d_k = self.hidden_size // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.hidden_size) for _ in range(3)])
        self.output = nn.Linear(self.hidden_size, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_attention_heads = h
        self.attention_head_size = self.d_k
        # for linears in self.linears:
        #     torch.nn.init.xavier_uniform(linears.weight)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, type_scores, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attentionWithTypeScores(query, key, value, type_scores, mask=mask,
                                               dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.output(x)


class SelfAttention(nn.Module):
    def __init__(self, head_num, hidden_size, dropout=.1):
        super(SelfAttention, self).__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, head_num))
        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_size / head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_in, k_in, v_in, attention_mask=None):
        mixed_query_layer = self.query(q_in)
        mixed_key_layer = self.key(k_in)
        mixed_value_layer = self.value(v_in)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if (attention_mask is not None):
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class TypeOneToManyAttention(nn.Module):
    def __init__(self, head_num, hidden_size, dropout=.1):
        super(TypeOneToManyAttention, self).__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, head_num))
        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_size // head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    # def transpose_for_scores3(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, q_in, k_in, v_in, type_score, attention_mask=None):
        mixed_query_layer = self.query(q_in)
        mixed_key_layer = self.key(k_in)
        mixed_value_layer = self.value(v_in)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch_size,head_num,max_len,head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if (attention_mask is not None):
            print(attention_scores.size())
            print(attention_mask.size())
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        type_attention_probs = attention_probs * type_score
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        type_attention_probs = self.dropout(type_attention_probs)  # ((batch_size,head_num,max_len,max_len))

        context_layer = torch.matmul(type_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).squeeze(-2)
        return context_layer


class SelfAttentivePooling(nn.Module):
    def __init__(self, hidden_size, entity_hidden_size):
        super(SelfAttentivePooling, self).__init__()
        # batch_span_list:(batch_size,max_span_num,span_length)
        self.se = SelfAttentiveSpanExtractor(input_dim=hidden_size)
        self.activation = nn.GELU()
        self.projection_matrix = nn.Linear(hidden_size, entity_hidden_size)
        self.layer_norm = nn.LayerNorm(entity_hidden_size, eps=1e-12)

    def forward(self, hidden_state, span_indices):
        span_rep = self.se(hidden_state, span_indices)
        span_rep_act = self.activation(span_rep)
        span_rep_proj = self.projection_matrix(span_rep_act)
        outputs = self.layer_norm(span_rep_proj)
        return outputs


if __name__ == '__main__':
    hidden_state = torch.randn(2, 64, 768)
    attpooler = SelfAttentivePooling(768, 200)
    span_indices = torch.LongTensor([[(0, 2), (2, 4)], [(0, 2), (2, 4)]])
    outputs = attpooler(hidden_state, span_indices)

    knowledge_path = '../knowledge/CMEKG'
    kg_ckpts_path = knowledge_path + '/ckpts'
    entity_type_emb = np.load(kg_ckpts_path + '/TransE_l2_schema_0/schema_TransE_l2_entity.npy')
    entity_type_emb = np.expand_dims(entity_type_emb, 0).repeat(2, axis=0)
    entity_type_emb = torch.from_numpy(entity_type_emb)
    att_layer = BasicAttention(200, 20, 20, 20, 20)
    type_emb = att_layer(outputs, entity_type_emb, entity_type_emb)
    result = torch.cat((outputs, type_emb), dim=-1)
    print(result)

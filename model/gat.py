"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl import load_graphs
from dgl.nn.pytorch import GATConv
import numpy as np
from model import SelfAttention, SelfAttentivePooling
from model.attention import BasicAttention


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class GAT1(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 activation=None,
                 residual=False):
        super(GAT1, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None, allow_zero_in_degree=True))

    def forward(self, g, inputs):
        h = inputs
        h = self.gat_layers[0](g, h).mean(1)
        return h


class GATWITHTYPE(nn.Module):
    def __init__(self, gat_layer, att_layer, graph_path, device, ent_with_type_size):
        super(GATWITHTYPE, self).__init__()
        self.gat_layer = gat_layer
        self.att_layer = att_layer
        self.graph_path = graph_path
        self.device = device
        self.ent_with_type_size = ent_with_type_size

    def forward(self, g_list, span_list, query, seq_len):
        batch_list = []
        for i, (list, spans) in enumerate(zip(g_list, span_list)):
            if len(list) == 0:
                batch_list.append(th.zeros(seq_len, self.ent_with_type_size, device=self.device))
                continue
            glist, label_dict = load_graphs(self.graph_path + "/data.bin", list)
            glist = dgl.batch(glist).to(self.device)
            input_feature = th.cat((glist.ndata['node_feature'], glist.ndata['node_type_feature']), -1)
            glist.ndata['ft'] = self.gat_layer(glist, input_feature)
            glist = dgl.unbatch(glist)
            result_list = [th.zeros(spans[0][0], self.ent_with_type_size, device=self.device)]
            spans = spans[:len(list)]
            for j, (g, span) in enumerate(zip(glist, spans)):
                output = g.ndata['ft']
                ao = self.att_layer(query[i, j, :].unsqueeze(0), output, output)
                ao = ao.squeeze(0).squeeze(0).repeat(span[1] - span[0] + 1).reshape(span[1] - span[0] + 1, -1)
                result_list.append(ao)
                if j + 1 < len(spans):
                    result_list.append(
                        th.zeros(spans[j + 1][0] - span[1] - 1, self.ent_with_type_size, device=self.device))
            result_list.append(th.zeros(seq_len - 1 - span[1], self.ent_with_type_size, device=self.device))
            result = th.cat(result_list)
            batch_list.append(result)
        batch_result = th.stack(batch_list)
        return batch_result


class GraphEnhancingModule(nn.Module):
    def __init__(self, graph_path, kg_ckpts_path, device, hidden_size=768, entity_rep_size=400, entity_type_size=20):
        super(GraphEnhancingModule, self).__init__()
        self.attpooler = SelfAttentivePooling(hidden_size, entity_rep_size)
        self.att_layer1 = BasicAttention(entity_rep_size, entity_type_size, entity_type_size, entity_type_size,
                                         entity_type_size)
        self.ent_with_type_size = entity_rep_size + entity_type_size
        self.att_layer2 = BasicAttention(self.ent_with_type_size, self.ent_with_type_size, self.ent_with_type_size,
                                         self.ent_with_type_size, self.ent_with_type_size)
        self.gat_layer = GAT1(num_layers=1, in_dim=self.ent_with_type_size, num_hidden=self.ent_with_type_size,
                              heads=[1])
        self.gat_with_type = GATWITHTYPE(self.gat_layer, self.att_layer2, graph_path, device, self.ent_with_type_size)
        self.self_attention_layer = SelfAttention(head_num=12,
                                                  hidden_size=hidden_size + entity_rep_size + entity_type_size)
        self.entity_type_emb = np.load(kg_ckpts_path)
        self.device = device

    def forward(self, last_hidden_state, span_list, term_list, seq_len):
        # 自注意力求和
        # span_rep:(batch_size,num_span,400)
        span_rep = self.attpooler(last_hidden_state, span_list)
        # 实体类型预估
        entity_type_emb = np.expand_dims(self.entity_type_emb, 0).repeat(span_rep.shape[0], axis=0)
        entity_type_emb = th.from_numpy(entity_type_emb).to(self.device)
        span_type_emb = self.att_layer1(span_rep, entity_type_emb, entity_type_emb)
        query = th.cat((span_rep, span_type_emb), dim=-1)
        # 图注意力传递
        # graph_result:(batch_size,seq_len,420),enhanced_result:(batch_size,seq_len,1188)
        graph_result = self.gat_with_type(term_list, span_list, query, seq_len)
        enhanced_result = th.cat((last_hidden_state, graph_result), -1)
        # 自注意力知识融合
        infused_representation = self.self_attention_layer(enhanced_result, enhanced_result, enhanced_result)
        return infused_representation

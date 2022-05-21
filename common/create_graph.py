import json

import dgl
import torch as th
import numpy as np
from tqdm import tqdm
from model.attention import BasicAttention
from model.gat import GAT, GAT1
from common import read_dict, read_schema, write_line, set_seed
from dgl.data.utils import save_graphs, load_graphs


def create_INANDOUT_dict(read_path):
    triples = read_schema(read_path)
    in_dict = {}
    out_dict = {}
    for line in triples:
        if line[2] not in in_dict.keys():
            in_dict[line[2]] = []
        if line[0] not in out_dict.keys():
            out_dict[line[0]] = []
        out_dict[line[0]].append(line[2])
        in_dict[line[2]].append(line[0])
    return in_dict, out_dict


def create_graph(read_path, write_path):
    entity_list = list(ent2id.keys())
    in_dict, out_dict = create_INANDOUT_dict(read_path)
    graph_list = []
    graph_labels = {'labels': []}
    for entity in tqdm(entity_list):
        srt = []
        drt = []
        srt_node = []
        drt_node = []
        node_list = [entity]
        if entity in in_dict.keys():
            srt_node = in_dict[entity][:2]
            srt = [x for x in range(1, len(srt_node) + 1)]
            drt = [0] * len(srt_node)
        if entity in out_dict.keys():
            drt_node = out_dict[entity][:2]
            srt += [0] * len(drt_node)
            drt += [x for x in range(len(srt_node) + 1, len(srt_node) + len(drt_node) + 1)]
        node_list = node_list + srt_node + drt_node
        node_type_list = [int(type2id[ent2type[x]]) for x in node_list]
        nodeID_list = [int(ent2id[x]) for x in node_list]
        node_type_feature = entity_type_emb[node_type_list,]
        node_feature = entity_emb[nodeID_list,]
        g = dgl.graph((srt, drt))
        g = dgl.add_self_loop(g)
        g.ndata['node_feature'] = th.from_numpy(node_feature)
        g.ndata['node_type_feature'] = th.from_numpy(node_type_feature)
        graph_list.append(g)
        graph_labels['labels'].append(int(ent2id[entity]))
    graph_labels['labels'] = th.tensor(graph_labels['labels'])
    save_graphs(write_path, graph_list, graph_labels)


def gat(g_list, gat_layer, att_layer, span_list, query,graph_path):
    batch_list = []
    for i, (list, spans) in enumerate(zip(g_list, span_list)):
        glist, label_dict = load_graphs(graph_path + "./data.bin", list)
        glist = dgl.batch(glist)
        input_feature = th.cat((glist.ndata['node_feature'], glist.ndata['node_type_feature']), -1)
        glist.ndata['ft'] = gat_layer(glist, input_feature)
        glist = dgl.unbatch(glist)
        result_list = [th.zeros(spans[0][0], 420)]
        spans = spans[:len(list)]
        for j, (g, span) in enumerate(zip(glist, spans)):
            output = g.ndata['ft']
            ao = att_layer(query[i, j, :].unsqueeze(0), output, output)
            ao = ao.squeeze(0).squeeze(0).repeat(span[1] - span[0] + 1).reshape(span[1] - span[0] + 1, -1)
            result_list.append(ao)
            if j + 1 < len(spans):
                result_list.append(th.zeros(spans[j+1][0]-span[1]-1, 420))
        result_list.append(th.zeros(63-span[1], 420))
        result = th.cat(result_list)
        batch_list.append(result)
    batch_result = th.stack(batch_list)
    return batch_result


def create_dict(entity_list, write_path):
    for line in entity_list:
        write_line(line, write_path)



if __name__ == '__main__':
    knowledge_path = '../knowledge/CMeKG2'
    ckpts_path = knowledge_path + '/ckpts'
    dict_path = knowledge_path + '/train'
    schema_path = knowledge_path + '/train_schema'
    graph_path = knowledge_path + '/graph'
    # size:(73332,400),ndarray
    entity_emb = np.load(ckpts_path + '/TransE_l2_BASE_3/BASE_TransE_l2_entity.npy')
    # size:(12,20),ndarray
    entity_type_emb = np.load(ckpts_path + '/TransE_l2_schema_1/schema_TransE_l2_entity.npy')
    id2ent = read_dict(dict_path + '/entities.tsv')
    ent2id = {v: k for k, v in id2ent.items()}
    ent2type = {line[0]: line[2] for line in read_schema(dict_path + '/entity_type_new.csv')}
    id2type = read_dict(schema_path + '/entities.tsv')
    type2id = {v: k for k, v in id2type.items()}

    # set_seed(42)
    # dataset = KGDatasetUDDRaw(path=dict_path, name='cmekg', files=['triples.csv'], format='hrt')
    # src, etype_id, dst = dataset.train
    # coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
    #                            shape=[dataset.n_entities, dataset.n_entities])
    # # g = dgl.DGLGraph(coo, readonly=True, multigraph=None, sort_csr=True)
    # g = dgl.graph(coo)
    # g.edata['tid'] = F.tensor(etype_id, F.int32)
    # g.ndata['feat'] = entity_emb
    # sg = dgl.in_subgraph(g,[1000,432,543,435,32,34,5,43])
    # hg = dgl.graph(sg.edges())
    # nx.draw(hg.to_networkx(), with_labels=True)
    # plt.show()
    # create_graph(dict_path + '/triples.csv', graph_path + '/data.bin')
    # glist, label_dict = load_graphs(graph_path + "./data.bin")
    glist, label_dict = load_graphs(graph_path + "./data.bin", [0,4,7,8])
    # g_list = [[0, 1], [2, 4, 6, 7]]
    # span_list = [[(0, 1), (2, 5), (0, 0), (0, 0)], [(0, 2), (3, 5), (6, 7), (10, 11)]]
    # query = th.randn(2, 7, 420)
    # gat_layer = GAT1(num_layers=1, in_dim=420, num_hidden=420, heads=[1])
    # att_layer = BasicAttention(420, 420, 420, 420, 420)
    # gat(g_list, gat_layer, att_layer, span_list, query,graph_path)
    # create_dict(list(ent2id.keys()), dict_path + '/entity_dict.txt')

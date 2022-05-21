# -*- coding: utf-8 -*-
import pkuseg
from transformers import BertModel, BertTokenizerFast

from common import read_dict, read_schema
from model import GraphEnhancingModule
from model.GlobalPointer import GlobalPointer
import json
import torch
import numpy as np
from tqdm import  tqdm

bert_model_path = 'chinese-roberta-wwm-ext' #your RoBert_large path
save_model_path = './outputs/ent_model.pth'
knowledge_path = 'knowledge/CMeKG2'
kg_ckpts_path = knowledge_path + '/ckpts'
kg_dict_path = knowledge_path + '/train'
schema_path = knowledge_path + '/train_schema'
graph_path = knowledge_path + '/graph'
device = torch.device("cuda:0")

max_len = 256
ent2id, id2ent = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
seg = pkuseg.pkuseg(model_name='./knowledge/medicine', user_dict=kg_dict_path + '/entity_dict.txt')
id2entity = read_dict(kg_dict_path + '/entities.tsv')
entity2id = {v: k for k, v in id2entity.items()}
ent2type = {line[0]: line[2] for line in read_schema(kg_dict_path + '/entity_type_new.csv')}
wanted_type = ['症状', '部位', '疾病', '诊疗', '药物', '检查', '其他', '手术治疗', '其他治疗']
bert =BertModel.from_pretrained(bert_model_path)
graph_enhancing_layer = GraphEnhancingModule(graph_path,
                                             kg_ckpts_path + '/TransE_l2_schema_1/schema_TransE_l2_entity.npy', device,
                                             768, 400, 20).to(device)
model = GlobalPointer(bert, graph_enhancing_layer, 9, 64, 400, 20).to(device)  # 9个实体类型

model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()


def span_segment(text, start_mapping, end_mapping):
    span_list = []
    term_list = []
    cut_list = seg.cut(text.lower())
    start = 0
    for term in cut_list:
        end = start + len(term) - 1
        if start in start_mapping and end in end_mapping:
            start_l = start_mapping[start]
            end = end_mapping[end]
            span = (start_l, end)
            if term in entity2id.keys() and ent2type[term] in wanted_type and span[-1] < max_len:
                span_list.append(span)
                term_list.append(int(entity2id[term]))
                # term_list.append(term)
        start = start + len(term)
    if len(span_list) == 0:
        span_list.append((0, 0))
    return span_list, term_list

def NER_RELATION(text, tokenizer, ner_model,  max_len=256):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    span_list, term_list = span_segment(text, start_mapping, end_mapping)
    span_list = torch.LongTensor(span_list).unsqueeze(0).cuda()
    term_list = [term_list]
    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda().float()
    scores = ner_model(input_ids, attention_mask, token_type_ids, span_list, term_list, use_graph=True)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entity_name = ''
        for i in range(new_span[start][0], new_span[end][-1]+1):
            entity_name += text[i]
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l], "entity":entity_name})

    return {"text":text, "entities":entities}

if __name__ == '__main__':
    all_ = []
    for d in tqdm(json.load(open('datasets/CMeEE/CMeEE_test.json', encoding='utf-8'))):
        all_.append(NER_RELATION(d["text"], tokenizer=tokenizer, ner_model=model))
    json.dump(
        all_,
        open('./outputs/CMeEE_test.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )
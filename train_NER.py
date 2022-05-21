# -*- coding: utf-8 -*-
from common import set_seed, read_dict, read_schema
from data.data_loader import ENTDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
from model.GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm
import random
import os
import numpy as np
import torch
import pkuseg
from model import GraphEnhancingModule

set_seed(42)

bert_model_path = './chinese-roberta-wwm-ext'  # RoBert_large 路径
train_cme_path = 'datasets/CMeEE/CMeEE_train.json'  # CMeEE 训练集
eval_cme_path = 'datasets/CMeEE/CMeEE_dev.json'  # CMeEE 测试集
knowledge_path = 'knowledge/CMeKG2'
kg_ckpts_path = knowledge_path + '/ckpts'
kg_dict_path = knowledge_path + '/train'
schema_path = knowledge_path + '/train_schema'
graph_path = knowledge_path + '/graph'
device = torch.device("cuda:0")
BATCH_SIZE = 4

ENT_CLS_NUM = 9

# tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)
seg = pkuseg.pkuseg(model_name='./knowledge/medicine', user_dict=kg_dict_path + '/entity_dict.txt')
id2ent = read_dict(kg_dict_path + '/entities.tsv')
ent2id = {v: k for k, v in id2ent.items()}
ent2type = {line[0]: line[2] for line in read_schema(kg_dict_path + '/entity_type_new.csv')}
wanted_type = ['症状', '部位', '疾病', '诊疗', '药物', '检查', '其他', '手术治疗', '其他治疗']

# train_data and val_data
ner_train = ENTDataset(load_data(train_cme_path), tokenizer=tokenizer, segmentor=seg, entity_dict=ent2id,
                       ent2type=ent2type, wanted_type=wanted_type)
ner_loader_train = DataLoader(ner_train, batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=False)
ner_evl = ENTDataset(load_data(eval_cme_path), tokenizer=tokenizer, segmentor=seg, entity_dict=ent2id,
                     ent2type=ent2type, wanted_type=wanted_type)
ner_loader_evl = DataLoader(ner_evl, batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False)

# GP MODEL
bert = BertModel.from_pretrained(bert_model_path)
graph_enhancing_layer = GraphEnhancingModule(graph_path,
                                             kg_ckpts_path + '/TransE_l2_schema_1/schema_TransE_l2_entity.npy', device,
                                             768, 400, 20).to(device)
model = GlobalPointer(bert, graph_enhancing_layer, ENT_CLS_NUM, 64, 400, 20).to(device)  # 9个实体类型
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_pred, y_true):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_pred, y_true.float())
    return loss


metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, span_list, term_list = batch
        input_ids, attention_mask, segment_ids, labels, span_list = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device), span_list.to(device)
        logits = model(input_ids, attention_mask, segment_ids, span_list, term_list, use_graph=True)
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels.float())
        total_loss += loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        print("trian_loss:", avg_loss, "\t train_f1:", avg_f1)

    with torch.no_grad():
        total_f1_, total_precision_, total_recall_ = 0., 0., 0.
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, span_list, term_list = batch
            input_ids, attention_mask, segment_ids, labels, span_list = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device), span_list.to(device)
            logits = model(input_ids, attention_mask, segment_ids, span_list, term_list, use_graph=True)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))
        t = model.state_dict()
        if avg_f1 > max_f:
            t = model.state_dict()
            torch.save(model.state_dict(), './outputs/ent_model.pth'.format(eo))
            max_f = avg_f1
        model.train()

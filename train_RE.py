import time

import torch
import torch.nn.functional as F
from model import CasRel
from torch.utils.data import DataLoader
from data.data_loader import REDataset, collate_fn
from common import set_seed, read_dict, read_schema
from model import GraphEnhancingModule

device = 'cuda:0'
torch.set_num_threads(6)
batch_size = 1
epoch = 20
max_len = 128
model_path = 'chinese-roberta-wwm-ext'
# model_path = 'checkpoint/bert-base-chinese'
# model_path = 'checkpoint/mc_bert_base'
# model_path = 'checkpoint/PCL-MedBERT'
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

def get_loss(pred, gold, mask):
    pred = pred.squeeze(-1)
    loss = F.binary_cross_entropy(pred, gold.float(), reduction='none')  # 以向量形式返回loss
    if loss.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def evaluate(dataloader, model, config):
    total_loss = 0.
    with torch.no_grad():
        model.eval()
        for batch_index, (
        sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single) in enumerate(
                iter(dataloader)):
            batch_data = dict()
            batch_data['token_ids'] = sample
            batch_data['mask'] = mask
            batch_data['sub_start'] = sub_start_single
            batch_data['sub_end'] = sub_end_single
            pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = model(batch_data)
            sub_start_loss = get_loss(pred_sub_start, sub_start, mask)
            sub_end_loss = get_loss(pred_sub_end, sub_end, mask)
            obj_start_loss = get_loss(pred_obj_start, relation_start, mask)
            obj_end_loss = get_loss(pred_obj_end, relation_end, mask)
            loss = config['sub_weight'] * (sub_start_loss + sub_end_loss) + config['obj_weight'] * (
                    obj_start_loss + obj_end_loss)
            total_loss += loss
        avg_loss = total_loss / len(dataloader)
        model.train()
        return avg_loss


if __name__ == '__main__':
    config = {'mode': 'train', 'batch_size': batch_size, 'epoch': epoch, 'relation_types': 53, 'sub_weight': 1,
              'obj_weight': 1, \
              'max_len': max_len, 'model_path': model_path}
    train_path = 'data/CMeIE_train.json'
    dev_path = 'data/CMeIE_dev.json'
    train_data = REDataset(train_path, config)
    dev_data = REDataset(dev_path, config)
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    graph_enhancing_layer = GraphEnhancingModule(graph_path,
                                                 kg_ckpts_path + '/TransE_l2_schema_1/schema_TransE_l2_entity.npy',
                                                 device,
                                                 768, 400, 20).to(device)
    model = CasRel(config,graph_enhancing_layer).to(device)
    # model.load_state_dict(torch.load('params.pkl'))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch_index in range(config['epoch']):
        for batch_index, (
        sample, sub_start, sub_end, relation_start, relation_end, mask, sub_start_single, sub_end_single) in enumerate(
                iter(train_dataloader)):
            batch_data = dict()
            batch_data['token_ids'] = sample
            batch_data['mask'] = mask
            batch_data['sub_start'] = sub_start_single
            batch_data['sub_end'] = sub_end_single
            pred_sub_start, pred_sub_end, pred_obj_start, pred_obj_end = model(batch_data)
            sub_start_loss = get_loss(pred_sub_start, sub_start, mask)
            sub_end_loss = get_loss(pred_sub_end, sub_end, mask)
            obj_start_loss = get_loss(pred_obj_start, relation_start, mask)
            obj_end_loss = get_loss(pred_obj_end, relation_end, mask)
            loss = config['sub_weight'] * (sub_start_loss + sub_end_loss) + config['obj_weight'] * (
                        obj_start_loss + obj_end_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (total_batch % 100 == 0):
                dev_loss = evaluate(dev_dataloader, model, config)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    last_improve = total_batch
                    torch.save(model.state_dict(), 'params.pkl')
                    print('saved!!!,dev_best_loss: %f' % dev_best_loss)
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    print('end of training!!!!')

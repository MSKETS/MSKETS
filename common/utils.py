import csv
import json


# 读取文件，返回列表
import os
import random

import dgl
import numpy as np
import torch


def read_text(path):
    return [x.strip() for x in open(path, 'r', encoding='utf-8').readlines()]


# 读取json文件，返回列表
def read_lines(path,to_lower=False):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        list = []
        for line in lines:
            if to_lower:
                line = line.lower().replace(' ', '')
            data = json.loads(line)
            list.append(data)
        return list


# 单行写入txt,json
def write_line(content, path):
    with open(path, "a", encoding='utf-8') as f:
        f.write(content + '\n')


# 读取csv字典，返回字典
def read_dict(path):
    dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            dict[row[0]] = row[1]
    return dict


# 读取schema ,返回列表
def read_schema(path):
    list = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            list.append(row)
    return list


# 写入单行csv
def write_csv(content, path):
    with open(path, "a", encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(content)


def write_csv_list(content, path):
    with open(path, "a", encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(content)


# 读取json文件返回列表
def load_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    return samples


# 以列表的形式写入json文件
def write_json(content, path):
    with open(path, 'a', newline='', encoding='utf-8') as write_f:
        write_f.write(json.dumps(content, ensure_ascii=False) + '\n')


# 将一个列表,分成若干个大小为n的列表
def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]


# 将一个列表,分成n个大小均分的列表
def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        yield origin_list[i * cnt:(i + 1) * cnt]

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# -*- coding: utf-8 -*-
import csv
import json
import time

import requests
from utils import read_dict, write_json, write_line

queryApi = 'https://zstp.pcl.ac.cn:8002/knowledge'
types = {
    '药物': 'https://zstp.pcl.ac.cn:8002/load_tree/%E8%8D%AF%E7%89%A9',
    '疾病': 'https://zstp.pcl.ac.cn:8002/load_tree/%E7%96%BE%E7%97%85',
    '症状': 'https://zstp.pcl.ac.cn:8002/load_tree/%E7%97%87%E7%8A%B6',
    '诊疗': 'https://zstp.pcl.ac.cn:8002/load_tree/%E8%AF%8A%E7%96%97'
}
params = {"name": "氢氯噻嗪", "tree_type": "药物"}
knowledge_path = '../knowledge'
dict_path = knowledge_path + '/dict'
kg_path = knowledge_path + '/kgs'




def creat_drug_dict(dict_json):
    drug_dict = {}
    category_list = []
    for category in dict_json['categories']:
        if category["name"] not in ['关系', 'UMLS', 'ICD-10', 'OTC类型']:
            drug_dict[category["name"]] = []
        category_list.append(category["name"])
    for node in dict_json['node']:
        if node["label"] != "" and category_list[node["category"]] not in ['关系', 'UMLS', 'ICD-10', 'OTC类型']:
            drug_dict[category_list[node["category"]]].append(node["label"])
    return drug_dict


def single_case():
    drug_name = "缬沙坦氨氯地平片i"
    write_path = "../knowledge/kgs/drug1.json"
    params["name"] = drug_name
    res = requests.get(url=queryApi, params=params).content.decode('unicode-escape').lower()
    dict_json = json.loads(res, strict=False)
    dict_json = creat_drug_dict(dict_json)
    dict_json["中心词"] = drug_name
    with open(write_path, 'w') as write_f:
        write_f.write(json.dumps(dict_json, indent=4, ensure_ascii=False))


def read_category(web_url,type):
    write_path = kg_path + "/category_{}.json".format(type)
    res = requests.get(url=web_url).content.decode('unicode-escape')
    dict_json = json.loads(res, strict=False)
    with open(write_path, 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(dict_json, indent=4, ensure_ascii=False))

def create_termnology_dict():
    for key, value in types.items():
        read_path = kg_path + '/category_{}.json'.format(key)
        with open(read_path, encoding='utf-8') as f:
            data = json.load(f)
            nodes = data['nodes']
            for node in nodes:
                if 'datarange' in node['icon']:
                    with open(dict_path + '/terminology_dict.csv', "a", encoding='utf-8', newline='') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow([node['name'], key])


def single_query(name, tree_type):
    params["name"] = name
    params['tree_type'] = tree_type
    res = requests.get(url=queryApi, params=params).content.decode('unicode-escape')
    try:
        dict_json = json.loads(res, strict=False)
    except Exception as e:
        write_line(name, kg_path + '/error.txt')
        write_line(str(e), kg_path + '/error.txt')
        write_line(res, kg_path + '/error.json')
        return 'error'
    else:
        return dict_json


def query_all_entities():
    print('start')
    time_start = time.time()  # 开始计时
    # 获取所有可查询实体
    entity_dict = read_dict(dict_path + '/111.csv')
    cnt = 0
    for name, tree_type in entity_dict.items():
        res = single_query(name, tree_type)
        if res != 'error':
            dict_json = creat_drug_dict(res)
            dict_json['中心词'] = name
            dict_json['entity_type'] = tree_type
            dict_json = json.dumps(dict_json, ensure_ascii=False).encode('utf-16', 'surrogatepass').decode('utf-16')
            write_line(dict_json, kg_path + '/graphs.json')
        cnt += 1
        if cnt % 500 == 0:
            time_processing = time.time()
            time_diff = time_processing - time_start
            print('{}:{},time cost:{}'.format(name, tree_type, time_diff))
            time.sleep(30)
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')




if __name__ == "__main__":
    query_all_entities()




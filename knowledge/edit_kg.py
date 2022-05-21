import csv
import random
import tqdm
from common import read_text, write_csv, read_lines, read_dict, read_schema, write_csv_list

knowledge_path = '.'
dict_path = knowledge_path + '/dict'
kg_path = knowledge_path + '/kgs'
schemas_path = '../datasets/CMeIE/53_schemas.json'
synonym_keys = {
    '同义词': ['通用名', '同义词', '英文名称'],
    '就诊科室': ['所属科室'],
    '放射治疗': ['放化疗治疗'],
    '化疗': ['化疗方案'],
    '相关（转化）': ['相关转化'],
    '相关（导致）': ['相关导致'],
    '相关（症状）': ['相关疾病'],
    '临床表现': ['临床症状及体征'],
    '治疗后症状': ['治疗引起的并发症'],
}
reversed_synonym_keys = {'通用名': '同义词', '同义词': '同义词', '英文名称': '同义词', '所属科室': '就诊科室', '放化疗治疗': '放射治疗', '化疗方案': '化疗',
                         '相关转化': '相关（转化）', '相关导致': '相关（导致）', '相关疾病': '相关（症状）', '临床症状及体征': '临床表现',
                         '治疗引起的并发症': '治疗后症状'}

wanted_keys = {
    '疾病': ['预防', '阶段', '所属科室', '辅助治疗', '化疗方案', '放化疗治疗', '手术治疗', '实验室检查', '影像学检查', '辅助检查', '组织学检查', '内窥镜检查', '筛查',
           '多发群体', '发病率', '发病年龄', '多发地区', '发病性别倾向', '死亡率', '多发季节', '传播途径', '通用名', '同义词', '英文名称', '并发症', '病理分型', '相关导致',
           '鉴别诊断', '相关转化', '相关疾病', '临床症状及体征', '治疗引起的并发症', '侵及周围组织或转移的症状', '病因', '高危因素', '风险评估因素', '病史', '遗传因素', '发病机制',
           '病理生理', '药物治疗', '发病部位', '转移部位', '外侵部位', '预后状况', '预后生存率', '检查', '治疗方案'],
    '症状': ['英文名称','所属科室', '发病部位', '检查', '相关症状'],
    '药物': ['通用名', '同义词', '英文名称', '药物相互作用', '不良反应', '适应症', '适应证'],
    '诊疗': ['英文名称', '子类']
}


def get_triples(read_graph_path, read_schema_path, write_entity_path, write_type_path):
    graph_list = read_lines(read_graph_path, to_lower=True)
    schema_list = read_schema(read_schema_path)
    schema_dict = {}
    for line in schema_list:
        schema_dict[line[0] + line[1]] = line[2]
    for graph in graph_list:
        head = graph['中心词'].replace('\n', '').replace('。', '').strip(' ')
        head_type = graph['entity_type']
        write_csv([head, '实体类型', head_type], write_type_path)
        for key, values in graph.items():
            if key in wanted_keys[head_type]:
                for tail in values:
                    tail = tail.replace('\n', '').replace('。', '').rstrip(',;')
                    if key in reversed_synonym_keys.keys():
                        key = reversed_synonym_keys[key]
                    tail = tail.split(',')
                    for item in tail:
                        # 舍弃长度大于20,小于2的尾实体
                        item = item.rstrip(',;').strip(' ')
                        if 20 >= len(item) >= 2 or key == '同义词':
                            write_csv([head, key, item], write_entity_path)
                            write_csv([item, '实体类型', schema_dict[head_type + key]], write_type_path)


def create_schema(read_path, write_schema_path):
    schemas_list = read_lines(read_path)
    for schema in schemas_list:
        head = schema['subject_type']
        relation = schema['predicate']
        tail = schema['object_type']
        write_csv([head, relation, tail], write_schema_path)


def create_wanted_keys(read_schema_path):
    schema_list = read_schema(read_schema_path)
    wanted_keys = {}
    for line in schema_list:
        if line[0] not in wanted_keys.keys():
            wanted_keys[line[0]] = []
        if line[1] in synonym_keys.keys():
            wanted_keys[line[0]] += synonym_keys[line[1]]
        else:
            wanted_keys[line[0]].append(line[1])
    print(wanted_keys)


def reverse_synonym_keys():
    reversed_synonym_keys = {}
    for key, value in synonym_keys.items():
        if len(value) == 1:
            reversed_synonym_keys[value[0]] = key
        else:
            for x in value:
                reversed_synonym_keys[x] = key
    print(reversed_synonym_keys)


def split_dataset(read_path, write_path, train_ratio, dev_ratio):
    triple_list = read_schema(read_path)
    random.shuffle(triple_list)
    write_csv_list(triple_list[0:int(len(triple_list) * train_ratio)], write_path + '/train.tsv')
    write_csv_list(triple_list[int(len(triple_list) * train_ratio):int(len(triple_list) * train_ratio) + int(
        len(triple_list) * dev_ratio)], write_path + '/valid.tsv')
    write_csv_list(triple_list[int(len(triple_list) * train_ratio) + int(len(triple_list) * dev_ratio):],
                   write_path + '/test.tsv')


def compress_entity_type(read_path, write_path):
    entity_type_list = read_schema(read_path)
    entity_type_dict = {}
    for entity_type in entity_type_list:
        if entity_type[0] not in entity_type_dict.keys():
            entity_type_dict[entity_type[0]] = entity_type[2]
    for key, value in entity_type_dict.items():
        write_csv([key, '实体类型', value], write_path)


if __name__ == "__main__":
    # get_triples(kg_path + '/graphs.json', kg_path + '/schema.csv', kg_path + '/triples.csv',
    #             kg_path + '/entity_type.csv')
    # create_schema(schemas_path, kg_path + '/schema.csv')
    # create_wanted_keys(kg_path + '/schema.csv')
    # reverse_synonym_keys()
    split_dataset('CMeKG2/train/triples.csv', './CMeKG2/train', 0.8, 0.1)
    # compress_entity_type(kg_path + '/entity_type.csv', kg_path + '/entity_type_new.csv')

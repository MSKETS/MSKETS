import csv
import json

knowledge_path = '.'
dict_path = knowledge_path + '/dict'
kg_path = knowledge_path + '/kgs'


def read_and_write(read_path, write_path):
    list = []
    with open(read_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            list.append([int(row[0]), row[1]])

    with open(write_path, "a") as f:
        writer = csv.writer(f, delimiter='\t')
        for i in list:
            writer.writerow(i)


def dealt_with_error(read_path):
    with open(read_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            print(line)




if __name__ == '__main__':
    dealt_with_error(kg_path + '/11.json')

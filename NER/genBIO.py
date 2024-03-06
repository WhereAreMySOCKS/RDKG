import json
import csv

import pandas as pd

file_path = "../rdkg/data/dataset.json"
stop = [" ", "：", "\"", "\n", "。", "、", "，", ",", "“", "”"]


def create_label(data_list, label_dict):
    re = []
    for data in data_list:
        text = data['text']
        sequence_labeling,sequence_labeling_idx = ['O'] * len(text),[0] * len(text)
        entity_result = data['entity_result']
        for entity in entity_result:
            entity_type_B = "B-"+entity['type']
            entity_type_I = "I-" + entity['type']
            if entity_type_B not in label_dict:
                label_dict[entity_type_B] = len(label_dict)
            if entity_type_I not in label_dict:
                label_dict[entity_type_I] = len(label_dict)
            start_idx = entity['startIndex']
            end_idx = entity['endIndex']
            sequence_labeling[start_idx] = entity_type_B
            sequence_labeling_idx[start_idx] = label_dict[entity_type_B]
            for i in range(start_idx + 1, end_idx):
                sequence_labeling[i] = entity_type_I
                sequence_labeling_idx[i] = label_dict[entity_type_I]
        re.append([list(text), str(sequence_labeling), str(sequence_labeling_idx),text,len(sequence_labeling)])
    return re

def write_csv(file_path, data):
    df = pd.DataFrame(data,columns=['sen', 'label_decode', 'label', 'raw_sen', 'length'])
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    label_dict = {  "start": 0,"end": 1,"pad": 2,"O":3}
    with open(file_path, "r", encoding="utf-8") as file:
        all_data_list = [json.loads(line.strip()) for line in file]
        labels = create_label(all_data_list, label_dict)
        dev = labels[int(0.9 * len(all_data_list)):]
        train = labels[:int(0.8 * len(all_data_list))]
        test = labels[int(0.8 * len(all_data_list)):int(0.9 * len(all_data_list))]
        write_csv("data/train_BIO.csv",train)
        write_csv("data/test_BIO.csv",test)
        write_csv("data/dev_BIO.csv",dev)
    with open("data/label_2_id.json", 'w',encoding="utf-8") as json_file:
        json.dump(label_dict, json_file)

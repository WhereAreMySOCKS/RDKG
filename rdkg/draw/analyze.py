"""
分析数据分布情况
"""
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = fm.findfont(fm.FontProperties(family='SimSun'))

def key_entity_distribution():
    _count = 0
    disease = []
    symptom = []
    surgery = []
    drugs = []
    with open('../data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            relation_result = dic['relation_result']
            for item in relation_result:
                if item['relation'] == '疾病症状':
                    disease.append(item['subject']['entity'])
                    symptom.append(item['object']['entity'])
                elif item['relation'] == '疾病手术':
                    disease.append(item['subject']['entity'])
                    surgery.append(item['object']['entity'])
                elif item['relation'] == '疾病药物':
                    disease.append(item['subject']['entity'])
                    drugs.append(item['object']['entity'])

    disease_dict, i_count = {}, 0
    for i in list(set(disease)):
        disease_dict.update({i: i_count})
        i_count += 1

    symptom_dict, j_count = {}, 0
    for j in list(set(symptom)):
        symptom_dict.update({j: j_count})
        j_count += 1

    surgery_dict, k_count = {}, 0
    for k in list(set(surgery)):
        surgery_dict.update({k: k_count})
        k_count += 1

    drugs_dict, l_count = {}, 0
    for l in list(set(drugs)):
        drugs_dict.update({l: l_count})
        l_count += 1

    _disease_count = [0 for _ in disease_dict.keys()]
    _symptom_count = [0 for _ in symptom_dict.keys()]
    _surgery_count = [0 for _ in surgery_dict.keys()]
    _drugs_count = [0 for _ in drugs_dict.keys()]

    for d in disease:
        _disease_count[disease_dict[d]] += 1

    for s in symptom:
        _symptom_count[symptom_dict[s]] += 1

    for d in surgery:
        _surgery_count[surgery_dict[d]] += 1

    for s in drugs:
        _drugs_count[drugs_dict[s]] += 1

    return _count, _disease_count, _symptom_count,_surgery_count,_drugs_count
def other_entity_distribution():
    other_entity_name = {}

    with open('../data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            entity_result = dic['entity_result']
            for i in entity_result:
                entity = i['entity']
                type =  i['type']
                if not other_entity_name.get(type):
                    other_entity_name[type] = [entity]
                else:
                    if entity not in other_entity_name[type]:
                        other_entity_name[type].append(entity)

        for k,v in other_entity_name.items():
            print("{} is {} for example {}".format(k,len(v),v))
def relation_distribution():
    relation_name = {}

    with open('../data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            relation_result = dic['relation_result']
            for item in relation_result:
                relation = item['relation']
                if not relation_name.get(relation):
                    relation_name[relation] = 1
                else:
                    relation_name[relation] += 1
    for k, v in relation_name.items():
        print("{} is {} ".format(k, v))
    print("relation num is {}".format(len(relation_name)))
    return relation_name
def draw_bar_input_ls(lst, title=None):
    plt.cla()
    plt.bar(range(len(lst)), lst)
    plt.title(title)
    plt.savefig(title, dpi=600)  # 保存为高清图片

    plt.show()
def draw_bar_input_dict(dct,title=None):
    plt.bar(dct.keys(), dct.values())
    plt.xticks([])  # 不显示横坐标
    plt.title(title)
    plt.savefig('bar_chart.png', dpi=300)  # 保存为高清图片


if __name__ == '__main__':
    count, disease_count, symptom_count,surgery_count,drugs_count = key_entity_distribution()
    print("疾病种类共 {},症状种类共 {},手术种类共 {},药物种类共 {}".format(len(disease_count),len(symptom_count),len(surgery_count),len(drugs_count)))
    draw_bar_input_ls(disease_count, "disease distribution")
    draw_bar_input_ls(symptom_count, "symptom distribution")
    draw_bar_input_ls(surgery_count, "surgery distribution")
    draw_bar_input_ls(drugs_count, "drugs distribution")
    # other_entity_distribution()
    # draw_bar_input_dict(relation_distribution())
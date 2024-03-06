import argparse
import jieba
import json
from tqdm import tqdm
from rdkg.src.data_utils import AierEyeDataset
from utils import *
import os
from rdkg.model.knowledge_graph import KnowledgeGraph

# Entities
HAVE_DISEASE = '疾病有'
HAVE_SYMPTOM = '症状有'
WORD = 'word'
SURGERY = '手术'
DRUG = '药物有'
ENTITIES = [HAVE_DISEASE, HAVE_SYMPTOM, SURGERY, DRUG]

# Attributes
POS_EXAM = '检查结果阳性'
NO_DISEASE = '疾病无'
NO_SYMPTOM = '症状无'
CAUSE = '诱因'

# Relations
DISEASE_SYMPTOM = '疾病症状'
DISEASE_SURGERY = '疾病手术'
DISEASE_DRUG = '疾病药物'
RELATED_SYMPTOM = '相关症状'
RELATED_DISEASE = '相关疾病'

ATTRIBUTES = [NO_DISEASE, NO_SYMPTOM, CAUSE, POS_EXAM]
RELATIONS = [DISEASE_SYMPTOM, DISEASE_SURGERY, DISEASE_DRUG, RELATED_SYMPTOM, RELATED_DISEASE]
raw_data = []


def tf_idf(vocab, reviews, word_tfidf_threshold):
    review_tfidf = compute_tfidf_fast(vocab, reviews)
    all_removed_words = []
    all_remained_words = []
    for rid, data in enumerate(reviews):
        doc_tfidf = review_tfidf[rid].toarray()[0]
        remained_words = [wid for wid in set(data) if doc_tfidf[wid] >= word_tfidf_threshold]

        removed_words = set(data).difference(remained_words)  # only for visualize
        removed_words = [vocab[wid] for wid in removed_words]
        _remained_words = [vocab[wid] for wid in remained_words]
        all_removed_words.append(removed_words)
        all_remained_words.append(_remained_words)
    return all_remained_words


def text_process(text):
    """
      对数据主诉部分进行预处理，去除停用词,载入词典。
      保存词典为dictionary/word.txt
    """
    #   疾病名称不进行分词
    jieba.load_userdict('../data/disease_dict.txt')
    stopwords = [line.strip() for line in open('../data/stopwords.txt', encoding='utf-8').readlines()]
    text_clean = []
    words = [jieba.lcut(t.replace('\n', '')) for t in text]
    #   去除停用词
    for wd in tqdm(words):
        tmp = []
        for w in wd:
            if w not in stopwords:
                tmp.append(w)
        text_clean.append(tmp)
    uni_words = sorted(set([j for i in text_clean for j in i]))
    entities2int = {}
    for i in range(len(uni_words)):
        entities2int['%s' % uni_words[i]] = i
    entities2int[''] = 0
    int_text = []
    #   将clean的主诉转换成index
    for i in text_clean:
        tmp = []
        for j in i:
            tmp.append(entities2int[j])
        int_text.append(tmp)
    #   使用tf-df去除高频词
    remain_word = tf_idf(list(entities2int.keys()), int_text, 0.20)
    int_text = []
    for i in remain_word:
        tmp = []
        for j in i:
            tmp.append(entities2int[j])
        int_text.append(tmp)
    return int_text, text_clean, entities2int


def entity_process(entity):
    if not entity:
        return []
    dic = {}
    entity = list(set(entity))
    for i in range(len(entity)):
        dic[entity[i]] = i
    return dic


def create_dictionary():
    """
       读取dataset.json文件，提取信息构造字典
    """
    useful_record = []
    entity = {'疾病有': [], '症状有': [], '手术': [], '药物有': []}
    chief_complaint = []
    textual_info = []
    # 记录每条记录中疾病和症状的对应关系
    with open('../data/dataset.json', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line)
            relation_result = dic['relation_result']
            is_record = False
            # for item in relation_result[::-1]:
            #     if item['relation'] == DISEASE_SYMPTOM:
            #         is_record = True
            #         break
            #   存在疾病症状关系后，再记录实体
            # if is_record:
            useful_record.append(relation_result)
            t, a = dic['text'].split('既往史：')[:2]

            # chief_complaint记录患者主诉，textual_info记录患者既往病史
            chief_complaint.append(t)
            textual_info.append(a)
            for item in relation_result:
                entity_type_s, entity_value_s = item['subject']['type'], item['subject']['entity']
                entity_type_v, entity_value_v = item['object']['type'], item['object']['entity']
                if entity_type_s in ENTITIES:
                    entity[entity_type_s].append(entity_value_s)
                if entity_type_v in ENTITIES:
                    entity[entity_type_v].append(entity_value_v)
    textual_info_dic = {}
    textual_info_set = list(set(textual_info))
    for i in range(len(textual_info_set)):
        textual_info_dic[textual_info_set[i]] = [i + 1]

    text, text_clean, text_dic = text_process(chief_complaint)
    all_text = []
    for t in text_clean:
        all_text.extend(t)
    dictionary = {}
    for k, v in entity.items():
        entity[k] = entity_process(v)
    diseases, symptoms, drugs, surgeries = [], [], [], []
    count = 0
    info = []
    for j in useful_record:
        info.append(textual_info_dic[textual_info[count]])
        count += 1
        dis, sym, drug, surgery = [], [], [], []
        for i in j:
            if i['relation'] == DISEASE_SYMPTOM:
                dis.append(entity['疾病有'][i['subject']['entity']])
                sym.append(entity['症状有'][i['object']['entity']])
            elif i['relation'] == DISEASE_DRUG:
                try:
                    # 标注不合理导致报错，忽略即可
                    drug.append(entity['药物有'][i['object']['entity']])
                except KeyError:
                    print(1)
            elif i['relation'] == DISEASE_SURGERY:
                try:
                    # 标注不合理导致报错，忽略即可
                    surgery.append(entity['手术'][i['object']['entity']])
                except KeyError:
                    print(1)

        diseases.append(list(set(dis)))
        symptoms.append(list(set(sym)))
        drugs.append(list(set(drug)))
        surgeries.append(list(set(surgery)))
    _len = len(diseases)
    dataset = []
    for i in range(_len):
        if not symptoms[i]:
            continue
        temp = [symptoms[i], diseases[i], text[i], info[i], [textual_info[i]]]
        dataset.append(temp)
    random.seed(1)
    random.shuffle(dataset)
    train_data, test_data = dataset[:int(0.8 * len(dataset))], dataset[int(0.8 * len(dataset)):]
    # pickle.dump(train_data, open('../data/processed_data/train.pkl', 'wb'))
    # pickle.dump(test_data, open('../data/processed_data/test.pkl', 'wb'))
    pickle.dump(gen_dataset(train_data), open(TMP_DIR[args.dataset] + '/train.pkl', 'wb'))
    pickle.dump(gen_dataset(test_data), open(TMP_DIR[args.dataset] + '/test.pkl', 'wb'))

    print('保存字典....')
    text_inverse = {}
    entity_inverse = {}
    for k, v in entity.items():
        new_dict = {}
        for key, value in v.items():
            new_dict[value] = key
        entity_inverse[k] = new_dict

    for key, value in dictionary.items():
        text_inverse[value] = key

    pickle.dump(text_inverse, open('../data/processed_data/dictionary.pkl', 'wb'))
    pickle.dump(entity['疾病有'], open('../data/processed_data/disease.pkl', 'wb'))
    pickle.dump(entity['症状有'], open('../data/processed_data/symptom.pkl', 'wb'))
    pickle.dump(entity['药物有'], open('../data/processed_data/drug.pkl', 'wb'))
    pickle.dump(entity['手术'], open('../data/processed_data/surgery.pkl', 'wb'))
    pickle.dump(text_dic, open('../data/processed_data/word.pkl', 'wb'))

    return diseases, symptoms, drugs, surgeries, text, info, entity


def create_relation(disease, symptom, drug, surgery, texts, info, entity):
    disease_len = len(entity['疾病有'])
    symptom_len = len(entity['症状有'])
    disease_symptom = [[] for i in range(disease_len)]
    disease_drug = [[] for i in range(disease_len)]
    disease_surgery = [[] for i in range(disease_len)]
    disease_texts = [[] for i in range(disease_len)]
    disease_disease = [[] for i in range(disease_len)]
    symptom_symptom = [[] for i in range(symptom_len)]
    for index in range(len(disease)):
        for d in range(len(disease[index])):
            #   d 代表疾病编号
            disease_symptom[disease[index][d]].extend(symptom[index])
            disease_drug[disease[index][d]].extend(drug[index])
            disease_surgery[disease[index][d]].extend(surgery[index])
            disease_texts[disease[index][d]].extend(texts[index])
            disease_disease[disease[index][d]].extend(disease[index])

    for index in range(len(symptom)):
        for d in range(len(symptom[index])):
            #   d 代表疾病编号
            symptom_symptom[symptom[index][d]].extend(symptom[index])

    # 创建知识图谱表示训练测试集
    disease_symptom_set =  [list(set(i)) for i in disease_symptom]
    disease_drug_set =  [list(set(i)) for i in disease_drug]
    disease_surgery_set =  [list(set(i)) for i in disease_surgery]
    disease_texts_set =  [list(set(i)) for i in disease_texts]
    disease_disease_set =  [list(set(i)) for i in disease_disease]
    symptom_symptom_set =  [list(set(i)) for i in symptom_symptom]

    print('保存关系....')
    pickle.dump(disease_symptom_set, open('../data/processed_data/disease_symptom.pkl', 'wb'))
    pickle.dump(disease_drug_set, open('../data/processed_data/disease_drug.pkl', 'wb'))
    pickle.dump(disease_surgery_set, open('../data/processed_data/disease_surgery.pkl', 'wb'))
    pickle.dump(disease_texts_set, open('../data/processed_data/disease_text.pkl', 'wb'))
    pickle.dump(disease_disease_set, open('../data/processed_data/disease_disease.pkl', 'wb'))
    pickle.dump(symptom_symptom_set, open('../data/processed_data/symptom_symptom.pkl', 'wb'))
    pickle.dump(info, open('../data/processed_data/textual_info.pkl', 'wb'))





    # pickle.dump(disease_symptom_set, open('../data/processed_data/disease_symptom.pkl', 'wb'))
    # pickle.dump(disease_drug_set, open('../data/processed_data/disease_drug.pkl', 'wb'))
    # pickle.dump(disease_surgery_set, open('../data/processed_data/disease_surgery.pkl', 'wb'))
    # pickle.dump(disease_texts_set, open('../data/processed_data/disease_text.pkl', 'wb'))
    # pickle.dump(disease_disease_set, open('../data/processed_data/disease_disease.pkl', 'wb'))
    # pickle.dump(symptom_symptom_set, open('../data/processed_data/symptom_symptom.pkl', 'wb'))
    # pickle.dump(info, open('../data/processed_data/textual_info.pkl', 'wb'))


    # print('保存关系....')
    # pickle.dump(disease_symptom_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/disease_symptom.pkl', 'wb'))
    # pickle.dump(disease_drug_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/disease_drug.pkl', 'wb'))
    # pickle.dump(disease_surgery_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/disease_surgery.pkl', 'wb'))
    # pickle.dump(disease_texts_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/disease_text.pkl', 'wb'))
    # pickle.dump(disease_disease_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/disease_disease.pkl', 'wb'))
    # pickle.dump(symptom_symptom_set[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/symptom_symptom.pkl', 'wb'))
    # pickle.dump(info[int(0.9 * len(disease_symptom_set)):], open('../data/KGTestData/textual_info.pkl', 'wb'))
def gen_dataset(data):
    """

    :param data: 多个疾病对应多个症状
    :return: 症状疾病的一对一关系
    """
    result = []
    for d in data:
        symptoms,disease, word_id,info_id,info = d
        for s in symptoms:
            for dis in disease:
                result.append([[s],[dis],word_id,info_id,info])
    return result




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=Aier_EYE)
    args = parser.parse_args()
    # ========== BEGIN ========== #
    d, s, dr, sur, text, textual_info, m_len = create_dictionary()
    create_relation(d, s, dr, sur, text, textual_info, m_len)
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = AierEyeDataset(DATASET_DIR[args.dataset])
    save_dataset(args.dataset, dataset)

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    kg = KnowledgeGraph(dataset, args.dataset)
    kg.compute_degrees()
    # check_test_path(args.dataset, kg)
    save_kg(args.dataset, kg)
    # =========== END =========== #

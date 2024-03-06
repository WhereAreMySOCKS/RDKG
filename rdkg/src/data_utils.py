from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
from easydict import EasyDict as edict
import random

from rdkg.src.utils import TMP_DIR


class AierEyeDataset(object):
    """This class is used to load raw_data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=0):
        self.data_dir = data_dir
        # if not self.data_dir.endswith('/'):
        #     self.data_dir += '/'
        if data_dir == 'Medical':
            self.train_file = 'medical_data/tmp/medical_data/' + set_name + '.pkl'
        else:
            self.train_file = TMP_DIR[self.data_dir] + set_name + '.pkl'
        self.load_entities()
        self.load_disease_relations()
        self.load_train_data()
        self.create_word_sampling_rate(word_sampling_rate)

        self.load_sympotms_relations()
        if self.data_dir == './raw_data/Aier_Eye':
            self.load_info()

    def _load_file(self, filename):
        with open(filename, 'rb') as f:
            load_data = pickle.load(f)
            #     因为dictionary和relation文件保存不一致，所以需要if
            if isinstance(load_data, list):
                return load_data
            elif isinstance(load_data, dict):
                return list(load_data.keys())

    def load_entities(self):
        """
        从picke文件中读取5个主要实体：
        ‘疾病有’，‘症状有’，‘手术’，‘药物有’，‘检查结果阳性’
         Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        if self.data_dir == 'Medical/':
            entity_files = edict(
                have_disease='medical_data/dictionary/疾病有.pkl',
                have_symptom='medical_data/dictionary/症状有.pkl',
                surgery='medical_data/dictionary/手术.pkl',
                drug='medical_data/dictionary/药物有.pkl',
                word='medical_data/dictionary/word.pkl',
            )
        else:
            entity_files = edict(
                have_disease='../data/processed_data/disease.pkl',
                have_symptom='../data/processed_data/symptom.pkl',
                surgery='../data/processed_data/surgery.pkl',
                drug='../data/processed_data/drug.pkl',
                word='../data/processed_data/word.pkl',
            )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print('Load', name, 'of size', len(vocab))

    def load_info(self):
        """
        从picke文件中读取既往病史（textual_info）：
        """

        vocab = self._load_file('../data/processed_data/info.pkl')
        vocab_size = len(vocab)
        setattr(self, 'info', edict(vocab=vocab, vocab_size=len(vocab)))
        print('Load attribute of size', vocab_size)

    def load_sympotms_relations(self):
        relation = edict(
            data=[],
            et_vocab=self.have_symptom.vocab,  # copy of brand, catgory ... 's vocab
            et_distrib=np.zeros(self.have_symptom.vocab_size)  # [1] means self.brand ..
        )
        if self.data_dir == 'Medical/':
            file_path = 'medical_data/relation/related_symptom_tail.pkl'
        else:
            file_path = '../data/processed_data/symptom_symptom.pkl'
        for line in self._load_file(file_path):
            knowledge = []
            for x in line:
                if x != -1:  # x = -1 代表无记录
                    knowledge.append(x)
                    relation.et_distrib[x] += 1
            relation.data.append(knowledge)
        setattr(self, 'related_symptom', relation)

    def load_train_data(self):

        train_data = []  # (have_symptom_idx, have_disease_idx, [word1_idx,...,wordn_idx])
        have_disease_distrib = np.zeros(self.have_disease.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.train_file):
            info_id = line[3]
            textual_info = line[4]
            chief_complaint_words = line[2]
            for i in range(len(line[0])):
                for j in range(len(line[1])):
                    train_data.append((line[0][i], line[1][j], chief_complaint_words, info_id, textual_info))
                    have_disease_distrib[line[1][j]] += 1
            for wi in chief_complaint_words:
                word_distrib[wi] += 1
            word_count += len(chief_complaint_words)
        self.train_data = edict(
            data=train_data,
            size=len(train_data),
            have_diseas_distrib=have_disease_distrib,
            have_disease_uniform_distrib=np.ones(self.have_disease.vocab_size),
            word_distrib=word_distrib,
            word_count=word_count,
            train_data_distrib=np.ones(len(train_data))  # set to 1 now
        )
        print('Load train_data of size', self.train_data.size, 'word count=', word_count)

    def load_disease_relations(self):
        """
            # Relations

                  Load 4 disease -> ? relations:
              - `disease_symptoms `: disease -> symptoms,
              - `disease_surgery `: disease -> surgery,
              - `disease_drugs `: disease -> drug,
              - `related_disease `: disease -> disease,

              Create member variable for each relation associated with following attributes:
              - `raw_data`: list of entity_tail indices (can be empty).
              - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
              - `et_distrib`: frequency of entity_tail vocab.
            """

        if self.data_dir == 'Medical/':
            product_relations = edict(
                disease_symptom=('medical_data/relation/disease_symptom_tail.pkl', self.have_symptom),
                disease_surgery=('medical_data/relation/disease_surgery_tail.pkl', self.surgery),
                disease_drug=('medical_data/relation/disease_drugs_tail.pkl', self.drug),
                related_disease=('medical_data/relation/related_disease_tail.pkl', self.have_disease),

            )
        else:
            product_relations = edict(
                disease_symptom=('../data/processed_data/disease_symptom.pkl', self.have_symptom),
                disease_surgery=('../data/processed_data/disease_surgery.pkl', self.surgery),
                disease_drug=('../data/processed_data/disease_drug.pkl', self.drug),
                related_disease=('../data/processed_data/disease_disease.pkl', self.have_disease),

            )

        for name in product_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `raw_data` variable saves list of entity_tail indices.
            # The i-th record of `raw_data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                data=[],
                et_vocab=product_relations[name][1].vocab,  # copy of brand, catgory ... 's vocab
                et_distrib=np.zeros(product_relations[name][1].vocab_size)  # [1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]):  # [0] means brand_p_b.pkl.gz ..
                knowledge = []
                for x in line:
                    knowledge.append(x)
                    relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', len(relation.data))

    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.train_data.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.train_data.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min(
                (np.sqrt(float(self.train_data.word_distrib[i]) / threshold) + 1) * threshold / float(
                    self.train_data.word_distrib[i]), 1.0)


class DataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_data_size = self.dataset.train_data.size
        self.have_disease_relations = ['disease_surgery', 'disease_drug', 'related_disease']
        self.related_symptom = ['related_symptom']
        # self.entity_dic = pickle.load(open('../data/processed_data/entity.pkl', 'rb'))
        self.text_dic = pickle.load(open('../data/processed_data/word.pkl', 'rb'))
        self.finished_word_num = 0
        self.reset()

    def reset(self):
        self.train_data_seq = np.random.permutation(self.train_data_size)
        self.cur_train_data_i = 0
        self.cur_word_i = 0
        self._has_next = True

    @property
    def get_batch(self):
        """
        每个batch中的数据为[症状，疾病，疾病手术，疾病药物，相关疾病，相关症状，既往史，既往史编号]
        """
        batch = []
        record_idx = self.train_data_seq[self.cur_train_data_i]
        have_symptom_idx, have_disease_idx, word, info_id, textual_info = self.dataset.train_data.data[record_idx]
        have_disease_knowledge = {pr: getattr(self.dataset, pr).data[have_disease_idx] for pr in
                                  self.have_disease_relations}
        related_symptom_knowledge = {pr: getattr(self.dataset, pr).data[have_symptom_idx] for pr in
                                     self.related_symptom}

        while len(batch) < self.batch_size:
            # 1) Sample the word
            word_idx = word[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                data = [have_symptom_idx, have_disease_idx, word_idx]
                for pr in self.have_disease_relations:
                    if len(have_disease_knowledge[pr]) <= 0:
                        data.append(0)
                    else:
                        data.append(random.choice(have_disease_knowledge[pr]))
                for pr in self.related_symptom:
                    if len(related_symptom_knowledge[pr]) <= 0:
                        data.append(0)
                    else:
                        data.append(random.choice(related_symptom_knowledge[pr]))
                data.append(textual_info[0])
                data.append(info_id[0])
                batch.append(data)

            # 2) Move to next word/train_data
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(word):
                self.cur_train_data_i += 1
                if self.cur_train_data_i >= self.train_data_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                train_data_idx = self.train_data_seq[self.cur_train_data_i]
                have_symptom_idx, have_disease_idx, word, info_id,textual_info = self.dataset.train_data.data[train_data_idx]
                have_disease_knowledge = {pr: getattr(self.dataset, pr).data[have_disease_idx] for pr in
                                          self.have_disease_relations}
                related_symptom_knowledge = {pr: getattr(self.dataset, pr).data[have_symptom_idx] for pr in
                                             self.related_symptom}

        random.shuffle(batch)
        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next

from __future__ import absolute_import, division, print_function
import numpy as np
import pickle
from easydict import EasyDict as edict
import random


class AierEyeDataset(object):
    """This class is used to load raw_data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=0):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        if data_dir == 'Medical':
            self.review_file = 'medical_data/tmp/medical_data/' + set_name + '.pkl'
        else:
            self.review_file = '../data/processed_data/' + set_name + '.pkl'
        self.load_entities()
        self.load_disease_relations()
        self.load_reviews()
        self.create_word_sampling_rate(word_sampling_rate)

        self.load_sympotms_relations()
        if self.data_dir == './raw_data/Aier_Eye':
            self.load_attribute()

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
                medicine='medical_data/dictionary/药物有.pkl',
                word='medical_data/dictionary/word.pkl',
            )
        else:
            entity_files = edict(
                have_disease='../data/processed_data/disease.pkl',
                have_symptom='../data/processed_data/symptom.pkl',
                surgery='../data/processed_data/surgery.pkl',
                medicine='../data/processed_data/drug.pkl',
                word='../data/processed_data/text.pkl',
            )
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print('Load', name, 'of size', len(vocab))

    def load_attribute(self):
        """
        从picke文件中读取3个主要属性：
        ‘疾病无’，‘症状无’，‘检查结果阳性’并全部保存为 self.attribute
        """

        vocab = self._load_file('data/dictionary/attribute.pkl')
        vocab_size = len(vocab)
        setattr(self, 'attribute', edict(vocab=vocab, vocab_size=len(vocab)))
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

    def load_reviews(self):
        """Load text from train/test raw_data files.
            Create member variable `review` associated with following attributes:
            - `raw_data`: list of tuples (user_idx, product_idx, [word_idx...]).
            - `size`: number of reviews.
            - `product_distrib`: product vocab frequency among all eviews.
            - `product_uniform_distrib`: product vocab frequency (all 1's)
            - `word_distrib`: word vocab frequency among all reviews.
            - `word_count`: number of words (including duplicates).
            - `review_distrib`: always 1.
            """

        review_data = []  # (have_symptom_idx, have_disease_idx, [word1_idx,...,wordn_idx])
        have_disease_distrib = np.zeros(self.have_disease.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            attr = line[3]
            word_indices = line[2]
            for i in range(len(line[0])):
                for j in range(len(line[1])):
                    review_data.append((line[0][i], line[1][j], word_indices, attr))
                    have_disease_distrib[line[1][j]] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.review = edict(
            data=review_data,
            size=len(review_data),
            have_diseas_distrib=have_disease_distrib,
            have_disease_uniform_distrib=np.ones(self.have_disease.vocab_size),
            word_distrib=word_distrib,
            word_count=word_count,
            review_distrib=np.ones(len(review_data))  # set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_disease_relations(self):
        """
            # Relations

                  Load 4 disease -> ? relations:
              - `disease_symptoms `: disease -> symptoms,
              - `disease_surgery `: disease -> surgery,
              - `disease_drugs `: disease -> medicine,
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
                disease_drug=('medical_data/relation/disease_drugs_tail.pkl', self.medicine),
                related_disease=('medical_data/relation/related_disease_tail.pkl', self.have_disease),

            )
        else:
            product_relations = edict(
                disease_symptom=('../data/processed_data/disease_symptom.pkl', self.have_symptom),
                disease_surgery=('../data/processed_data/disease_surgery.pkl', self.surgery),
                disease_drug=('../data/processed_data/disease_drug.pkl', self.medicine),
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
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min(
                (np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(
                    self.review.word_distrib[i]), 1.0)


class DataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.review_size = self.dataset.review.size
        self.have_disease_relations = ['disease_surgery', 'disease_drug', 'related_disease', ]
        self.related_symptom = ['related_symptom']
        self.entity_dic = pickle.load(open('../data/processed_data/entity.pkl', 'rb'))
        self.text_dic = pickle.load(open('../data/processed_data/text.pkl', 'rb'))
        self.finished_word_num = 0
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self.review_seq = np.random.permutation(self.review_size)
        self.cur_review_i = 0
        self.cur_word_i = 0
        self._has_next = True

    @property
    def get_batch(self):
        """Return a matrix of [batch_size x 6], where each row contains
        (have_symptom_idx, have_disease_idx, word_id, surgery_id, medicine_id
        , related_disease_idx,pos_exam,no_symptom,no_disease).
        """
        batch = []
        record_idx = self.review_seq[self.cur_review_i]
        have_symptom_idx, have_disease_idx, word, attribute_text = self.dataset.review.data[
            record_idx]
        have_disease_knowledge = {pr: getattr(self.dataset, pr).data[have_disease_idx] for pr in
                                  self.have_disease_relations}
        related_symptom_knowledge = {pr: getattr(self.dataset, pr).data[have_symptom_idx] for pr in
                                     self.related_symptom}
        have_disease_knowledge.update({'attribute_text': attribute_text, })

        while len(batch) < self.batch_size:
            # 1) Sample the word
            word_idx = word[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                data = [self.entity_dic['症状有'][have_symptom_idx], self.entity_dic['疾病有'][have_disease_idx],
                        self.text_dic[word_idx]]
                for pr in self.have_disease_relations:
                    if len(have_disease_knowledge[pr]) <= 0:
                        data.append(0)
                    else:
                        if pr == 'disease_surgery':
                            data.append(self.entity_dic['手术'][random.choice(have_disease_knowledge[pr])])
                        elif pr == 'disease_drug':
                            data.append(self.entity_dic['药物有'][random.choice(have_disease_knowledge[pr])])
                        elif pr == 'related_disease':
                            rc = list(set(have_disease_knowledge[pr]) - {have_disease_idx})
                            if rc:
                                data.append(self.entity_dic['疾病有'][random.choice(rc)])
                            else:
                                data.append(0)
                for pr in self.related_symptom:
                    rc = list(set(related_symptom_knowledge[pr]) - {have_symptom_idx})
                    if len(rc) > 0:
                        data.append(self.entity_dic['症状有'][random.choice(rc)])
                    else:
                        data.append(0)
                data.append(attribute_text)
                batch.append(data)

            # 2) Move to next word/review
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(word):
                self.cur_review_i += 1
                if self.cur_review_i >= self.review_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                review_idx = self.review_seq[self.cur_review_i]
                have_symptom_idx, have_disease_idx, word, attribute_text = self.dataset.review.data[review_idx]
                have_disease_knowledge = {pr: getattr(self.dataset, pr).data[have_disease_idx] for pr in
                                          self.have_disease_relations}
                have_disease_knowledge.update(
                    {'attribute_text': attribute_text})
        random.shuffle(batch)
        return batch

    def has_next(self):
        """Has next batch."""
        return self._has_next

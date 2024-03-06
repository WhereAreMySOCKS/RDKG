from rdkg.src.utils import *


class KnowledgeGraph(object):

    def __init__(self, dataset, dataset_name):
        self.G = dict()
        self._load_entities(dataset, dataset_name)
        self._load_related(dataset, dataset_name)
        self._load_reviews(dataset)
        self._load_knowledge(dataset, dataset_name)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset, dataset_name):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities(dataset_name):
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(entity, dataset_name)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_related(self, dataset, dataset_name):
        for relation in [RELATED_SYMPTOM]:
            data = getattr(dataset, relation).data
            num_edges = 0
            for head_id, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                eids = list(set(eids))
                for eid in eids:
                    et_type = get_entity_tail(HAVE_SYMPTOM, relation, dataset_name)
                    self._add_edge(HAVE_SYMPTOM, head_id, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _load_reviews(self, dataset):
        print('Load train_dataset...')
        #  Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        num_edges = 0
        all_removed_words = []
        all_remained_words = []
        for rid, data in enumerate(dataset.train_data.data):
            sym_id, dis_id, words = data[0], data[1], data[2]
            remained_words = [wid for wid in set(words)]
            removed_words = set(words).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            _remained_words = [vocab[wid] for wid in remained_words]
            all_removed_words.append(removed_words)
            all_remained_words.append(_remained_words)
            if len(remained_words) <= 0:
                continue
            # Add edges.
            try:
                self._add_edge(HAVE_SYMPTOM, sym_id, DISEASE_SYMPTOM, HAVE_DISEASE, dis_id)
                num_edges += 2
            except:
                print(1)
            for wid in remained_words:
                self._add_edge(HAVE_SYMPTOM, sym_id, MENTION, WORD, wid)
                self._add_edge(HAVE_DISEASE, dis_id, DESCRIBED_AS, WORD, wid)
                num_edges += 4
        print('Total {:d} words edges.'.format(num_edges))

    def _load_knowledge(self, dataset, dataset_name):
        for relation in [DISEASE_SYMPTOM, DISEASE_SURGERY, DISEASE_DRUG, RELATED_DISEASE]:
            data = getattr(dataset, relation).data
            num_edges = 0
            for did, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(HAVE_DISEASE, relation, dataset_name)
                    self._add_edge(HAVE_DISEASE, did, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_p_match, u_w_match):
        self.top_matches = {
            HAVE_SYMPTOM: u_u_match,
            HAVE_DISEASE: u_p_match,
            WORD: u_w_match,
        }

    def heuristic_search(self, sym_id, dis_id, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, sym_id)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 0:  # OK
            wids_u = set(_get(HAVE_SYMPTOM, sym_id, MENTION))  # SYMPTOM->MENTION->WORD
            wids_p = set(_get(HAVE_DISEASE, dis_id, DESCRIBED_AS))  # DISEASE->DESCRIBE->WORD
            intersect_nodes = wids_u.intersection(wids_p)
            paths = [(sym_id, x, dis_id) for x in intersect_nodes]
        elif pattern_id in [1, 2, 3, 11, 12, 13, 14, 15, 16, 17]:
            pids_u = set(_get(HAVE_SYMPTOM, sym_id, DISEASE_SYMPTOM))  # SYMPTOM-->DISEASE
            pids_u = pids_u.difference([dis_id])  # exclude target product
            nodes_p = set(_get(HAVE_DISEASE, dis_id, pattern[3][0]))  # DISEASE->relation->node2
            if pattern[2][1] == HAVE_SYMPTOM:
                nodes_p.difference([sym_id])
            for pid_u in pids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(HAVE_DISEASE, pid_u, relation))  # SYMPTOM-->DISEASE->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(sym_id, pid_u, x, dis_id) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 18:
            wids_u = set(_get(HAVE_SYMPTOM, sym_id, MENTION))  # SYMPTOM->MENTION->WORD
            uids_p = set(_get(HAVE_DISEASE, dis_id, DISEASE_SYMPTOM))  # DISEASE-->SYMPTOM
            uids_p = uids_p.difference([sym_id])  # exclude source SYMPTOM
            for uid_p in uids_p:
                wids_u_p = set(_get(HAVE_SYMPTOM, uid_p, MENTION))  # DISEASE-->SYMPTOM->MENTION->WORD
                intersect_nodes = wids_u.intersection(wids_u_p)
                tmp_paths = [(sym_id, x, uid_p, dis_id) for x in intersect_nodes]
                paths.extend(tmp_paths)
        return paths


def check_test_path(dataset_str, kg):
    # Check if there exists at least one path for any DISEASE-SYMPTOM in test set.
    test_user_products = pickle.load(open('../data/processed_data/test.pkl', 'rb'))
    for i in test_user_products:
        sym_id, dis_id = i[0], i[1]
        for j in sym_id:
            for x in dis_id:
                count = 0
                for pattern_id in [0, 1, 2, 3, 11, 12, 13, 14, 15, 16]:
                    tmp_path = kg.heuristic_search(j, x, pattern_id)
                    count += len(tmp_path)
                if count == 0:
                    print(x, dis_id)

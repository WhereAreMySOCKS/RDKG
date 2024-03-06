from __future__ import absolute_import, division, print_function
import os
import argparse
import pickle

import jieba
import torch.optim
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
from transe_model import KnowledgeEmbedding
from utils import *
from data_utils import DataLoader

logger = None


def test(args):
    # transe_embed = load_embed(args.dataset)
    # kg = pickle.load(open(TMP_DIR[args.dataset] + '/graph/dataset_test.pkl', 'rb'))
    disease_dic = pickle.load(open("../data/KGTestData/disease_disease.pkl",'rb'))
    # syms = [i[0][0] for i in test_data]
    # dis = [i[1][0] for i in test_data]
    print(disease_dic)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=Aier_EYE)
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--gpu', type=str, default='6', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--enhanced_type', type=str, default='none', help='model type. Including none,embedding,w2v')
    parser.add_argument('--embedding_type', type=str, default='TransE',help='Embedding model type,TransR,TransE or TransH')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)
    test(args)


if __name__ == '__main__':
    main()

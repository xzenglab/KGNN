'''
@Author: your name
@Date: 2019-12-20 19:02:25
@LastEditTime: 2020-04-14 19:46:00
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /matengfei/KGCN_Keras-master/config.py
'''
# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {
           'drug':os.path.join(RAW_DATA_DIR,'drug','train2id.txt'),
           'KEGG':os.path.join(RAW_DATA_DIR,'KEGG','train2id.txt'),
           'dti_drugbank':os.path.join(RAW_DATA_DIR,'dti_drugbank','train2id.txt')}
ITEM2ENTITY_FILE = {
                    'drug':os.path.join(RAW_DATA_DIR,'drug','entity2id.txt'),
                    'KEGG':os.path.join(RAW_DATA_DIR,'KEGG','entity2id.txt')
                    }
RATING_FILE = {
               'drug':os.path.join(RAW_DATA_DIR,'drug','approved_example.txt'),
               'KEGG':os.path.join(RAW_DATA_DIR,'KEGG','approved_example.txt')
               }
SEPARATOR = {'drug':'\t','KEGG':'\t'}
THRESHOLD = {'drug':4,'KEGG':4} 

NEIGHBOR_SIZE = {'drug':16,'KEGG':16}

USER_VOCAB_TEMPLATE = '{dataset}_user_vocab.pkl'
ITEM_VOCAB_TEMPLATE = '{dataset}_item_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
#RESULT_LOG='result.txt'
RESULT_LOG='result-db.txt'
#PERFORMANCE_LOG = 'kgcn_performance.log'
PERFORMANCE_LOG = 'KGNN_drugbank.log'
DRUGBANK_EXAMPLE='{dataset}_examples.npy'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 4 # neighbor sampling size
        self.embed_dim = 32  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 2e-2  # learning rate
        self.batch_size = 65536
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.optimizer = 'adam'

        self.drug_vocab_size = None
        self.item_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None

        self.exp_name = None
        self.model_name = None

        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='drug'
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3

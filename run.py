# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
from collections import defaultdict
sys.path.append(os.getcwd()) #add the env path
from sklearn.model_selection import train_test_split,StratifiedKFold
from main import train

from config import DRUGBANK_EXAMPLE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ITEM2ENTITY_FILE, KG_FILE, \
    RATING_FILE, USER_VOCAB_TEMPLATE, ITEM_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, THRESHOLD, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from utils import pickle_dump, format_filename,write_log,pickle_load

#[Errno 2] No such file or directory: './raw_data/movie/item_index2entity_id.txt'
def read_item2entity_file(file_path: str, item_vocab: dict, entity_vocab: dict):
    print(f'Logging Info - Reading item2entity file: {file_path}' )
    assert len(item_vocab) == 0 and len(entity_vocab) == 0
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if(count==0):
                count+=1
                continue
            item, entity = line.strip().split('\t')

            item_vocab[entity]=len(item_vocab) 
            entity_vocab[entity] = len(entity_vocab)

def read_example_file(file_path:str,separator:str,item_vocab:dict):
    print(f'Logging Info - Reading rating file: {file_path}')
    assert len(item_vocab)>0
    examples=[]
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            print('processing example: '+str(idx))
            d1,d2,flag=line.strip().split(separator)[:3]
            if d1 not in item_vocab or d2 not in item_vocab:
                continue
            if d1 in item_vocab and d2 in item_vocab:
                examples.append([item_vocab[d1],item_vocab[d2],int(flag)])
    
    examples_matrix=np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    X=examples_matrix[:,:2]
    y=examples_matrix[:,2:3]
    train_data_X, valid_data_X,train_y,val_y = train_test_split(X,y, test_size=0.2,stratify=y)
    train_data=np.c_[train_data_X,train_y]
    valid_data_X, test_data_X,val_y,test_y = train_test_split(valid_data_X,val_y, test_size=0.5)
    valid_data=np.c_[valid_data_X,val_y]
    test_data=np.c_[test_data_X,test_y]
    return examples_matrix


def read_rating_file(file_path: str, separator: str, threshold: int, user_vocab: dict,
                     item_vocab: dict):
    print(f'Logging Info - Reading rating file: {file_path}')

    assert len(user_vocab) == 0 and len(item_vocab) > 0
    user_pos_rating = defaultdict(set)
    user_neg_rating = defaultdict(set)
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            # if idx == 0:
            #     continue
            print('processing example: '+str(idx))
            user, item, rating = line.strip().split(separator)[:3]
            if item not in item_vocab:
                continue    # only consider items that has corresponding entities

            if rating == '1': #larger than 4 is positive sample 修改关系
                user_pos_rating[user].add(item_vocab[item])
            else:
                #user_neg_rating[user].add(item_vocab[item])
                user_neg_rating[user].add(item_vocab[item])#修改

    print('Logging Info - Converting rating file...')
    all_item_id_set = set(item_vocab.values())
    rating_data = []
    #user-item中1对多的关系
    for user, pos_item_id_set in user_pos_rating.items():
        user_vocab[user] = len(user_vocab) #为用户编号
        user_id = user_vocab[user]

        for item_id in pos_item_id_set:
            rating_data.append([user_id, item_id, 1])

        unwatched_set = all_item_id_set - pos_item_id_set #做集合差集
        if user in user_neg_rating:
            unwatched_set -= user_neg_rating[user]

        for item_id in np.random.choice(list(unwatched_set), size=len(pos_item_id_set),
                                        replace=False):
            rating_data.append([user_id, item_id, 0]) #做负样本处理，使用随机过程

    rating_matrix = np.array(rating_data)
    print(f'Logging Info - num of users: {len(user_vocab)}, num of items: {len(item_vocab)}')
    print(f'Logging Info - size of rating data: {rating_matrix.shape}')
    print(f'Logging Info - splitting rating data....')

    # train : dev : test = 6 : 2 : 2
    train_data, valid_data = train_test_split(rating_data, test_size=0.4)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5)

    return train_data, valid_data, test_data


def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if count==0:
                count+=1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph 使用双向关系
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        ##如果选取的接受域K的数量大于邻接点，则选取的时候
        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True
        )
        #随机获取图中节点的接受域
        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation


def process_data(dataset: str, neighbor_sample_size: int,K:int):
    user_vocab = {}
    item_vocab = {}
    entity_vocab = {}
    relation_vocab = {}
    #引用传递
    #read_item2entity_file(ITEM2ENTITY_FILE[dataset], item_vocab, entity_vocab)
    # train_data, dev_data, test_data = read_example_file(RATING_FILE[dataset], SEPARATOR[dataset],
    #                                                      item_vocab)
    if os.path.isfile(format_filename(PROCESSED_DATA_DIR, ITEM_VOCAB_TEMPLATE, dataset=dataset)):
        item_vocab=pickle_load(format_filename(PROCESSED_DATA_DIR, ITEM_VOCAB_TEMPLATE, dataset=dataset))
        entity_vocab=pickle_load(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset))
    else:
        read_item2entity_file(ITEM2ENTITY_FILE[dataset], item_vocab, entity_vocab)
    examples_file=format_filename(PROCESSED_DATA_DIR, DRUGBANK_EXAMPLE, dataset=dataset)
    if os.path.isfile(examples_file):
        examples=np.load(examples_file)
    else:
        examples = read_example_file(RATING_FILE[dataset], SEPARATOR[dataset],item_vocab)
        np.save(examples_file,examples)
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    
    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, USER_VOCAB_TEMPLATE, dataset=dataset),
            item_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ITEM_VOCAB_TEMPLATE, dataset=dataset),
                item_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_entity_file)
    cross_validation(K,examples,dataset,neighbor_sample_size)


def cross_validation(K_fold,examples,dataset,neighbor_sample_size):
    subsets=dict()
    n_subsets=int(len(examples)/K_fold)
    remain=set(range(0,len(examples)-1))
    for i in reversed(range(0,K_fold-1)):
        subsets[i]=random.sample(remain,n_subsets)
        remain=remain.difference(subsets[i])
    subsets[K_fold-1]=remain
    #aggregator_types=['sum','concat','neigh']
    aggregator_types=['sum','concat','neigh']
    for t in aggregator_types:
        count=1
        temp={'dataset':dataset,'aggregator_type':t,'avg_auc':0.0,'avg_acc':0.0,'avg_f1':0.0,'avg_aupr':0.0}
        for i in reversed(range(0,K_fold)):
            test_d=examples[list(subsets[i])]
            val_d,test_data=train_test_split(test_d,test_size=0.5)
            train_d=[]
            for j in range(0,K_fold):
                if i!=j:
                    train_d.extend(examples[list(subsets[j])])
            train_data=np.array(train_d)               
            train_log=train(
            kfold=count,
            dataset=dataset,
            train_d=train_data,
            dev_d=val_d,
            test_d=test_data,
            neighbor_sample_size=neighbor_sample_size,
            #embed_dim=32,
            embed_dim=32,
            #n_depth=2,
            n_depth=2,
            l2_weight=1e-7,
            lr=2e-2,
            #lr=5e-3,
            optimizer_type='adam',
            batch_size=512,
            aggregator_type=t,
            n_epoch=50,
            callbacks_to_add=['modelcheckpoint', 'earlystopping']
            )     
            count+=1
            temp['avg_auc']=temp['avg_auc']+train_log['test_auc']
            temp['avg_acc']=temp['avg_acc']+train_log['test_acc']
            temp['avg_f1']=temp['avg_f1']+train_log['test_f1']
            temp['avg_aupr']=temp['avg_aupr']+train_log['test_aupr']
        for key in temp:
            if key=='aggregator_type' or key=='dataset':
                continue
            temp[key]=temp[key]/K_fold
        write_log(format_filename(LOG_DIR, RESULT_LOG),temp,'a')
        print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')
   
if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    # process_data('movie', NEIGHBOR_SIZE['movie'])
    # process_data('music', NEIGHBOR_SIZE['music'])
    process_data('drug',NEIGHBOR_SIZE['drug'],5)
    




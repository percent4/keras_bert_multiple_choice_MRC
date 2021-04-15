# -*- coding: utf-8 -*-
# @Time : 2021/4/10 10:26
# @Author : Jclian91
# @File : params.py
# @Place : Yangpu, Shanghai
# 数据集配置
dataset = "RACE_middle"
train_file_path = "./data/{}/train.json".format(dataset)
dev_file_path = "./data/{}/dev.json".format(dataset)
test_file_path = "./data/{}/test.json".format(dataset)

# 模型配置
BERT_MODEL_DIR = "/nlp_group/nlp_pretrain_models/uncased_L-12_H-768_A-12"
VOCAB_FILE_PATH = "{}/vocab.txt".format(BERT_MODEL_DIR)
CONFIG_FILE_PATH = "{}/bert_config.json".format(BERT_MODEL_DIR)
CHECKPOINT_FILE_PATH = "{}/bert_model.ckpt".format(BERT_MODEL_DIR)

# 模型参数配置
NUM_CHOICES = 4
EPOCH = 8
BATCH_SIZE = 5
MAX_SEQ_LENGTH = 384
WARMUP_RATION = 0.1

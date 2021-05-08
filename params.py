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
BERT_MODEL_DIR = "/nlp_group/nlp_pretrain_models/wwm_uncased_L-24_H-1024_A-16"
VOCAB_FILE_PATH = "{}/vocab.txt".format(BERT_MODEL_DIR)
CONFIG_FILE_PATH = "{}/bert_config.json".format(BERT_MODEL_DIR)
CHECKPOINT_FILE_PATH = "{}/bert_model.ckpt".format(BERT_MODEL_DIR)

# 模型参数配置
NUM_CHOICES = 4
EPOCH = 20
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 320
WARMUP_RATION = 0.1

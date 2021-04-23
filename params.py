# -*- coding: utf-8 -*-
# @Time : 2021/4/10 10:26
# @Author : Jclian91
# @File : params.py
# @Place : Yangpu, Shanghai
# 数据集配置
dataset = "haihua_competition"
train_file_path = "./data/{}/new_train.json".format(dataset)
dev_file_path = "./data/{}/new_dev.json".format(dataset)
test_file_path = "./data/{}/test.json".format(dataset)

# 模型配置
BERT_MODEL_DIR = "/nlp_group/nlp_pretrain_models/chinese_RoBERTa_wwm_ext"
VOCAB_FILE_PATH = "{}/vocab.txt".format(BERT_MODEL_DIR)
CONFIG_FILE_PATH = "{}/bert_config.json".format(BERT_MODEL_DIR)
CHECKPOINT_FILE_PATH = "{}/bert_model.ckpt".format(BERT_MODEL_DIR)

# 模型参数配置
NUM_CHOICES = 4
EPOCH = 10
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 510
WARMUP_RATION = 0.1

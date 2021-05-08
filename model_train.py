# -*- coding: utf-8 -*-
# @Time : 2021/4/3 17:37
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import numpy as np
from keras_bert import Tokenizer
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras_bert import AdamWarmup, calc_train_steps
from tensorflow.python import math_ops, array_ops

from load_data import train_samples, dev_samples
# from zh_load_data import train_samples, dev_samples
from model import SimpleMultiChoiceMRC
from params import (dataset,
                    VOCAB_FILE_PATH,
                    BATCH_SIZE,
                    CHECKPOINT_FILE_PATH,
                    CONFIG_FILE_PATH,
                    NUM_CHOICES,
                    WARMUP_RATION,
                    EPOCH,
                    MAX_SEQ_LENGTH
                    )

token_dict = {}
with open(VOCAB_FILE_PATH, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict, cased=False)


class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            # np.random.shuffle(idxs)
            X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
            X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
            Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))
            i = 0
            for j in idxs:
                sample = self.data[j]
                Y[i % self.batch_size, sample.correct_answer] = 1
                for choice_num, answer in enumerate(sample.answers):
                    x1, x2 = tokenizer.encode(first=sample.article, second=sample.question+answer, max_len=MAX_SEQ_LENGTH)
                    X1[i % self.batch_size, choice_num, :] = x1
                    X2[i % self.batch_size, choice_num, :] = x2

                if ((i+1) % self.batch_size == 0) or j == idxs[-1]:
                    yield [X1, X2], Y
                    X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
                    X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_SEQ_LENGTH))
                    Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))

                i += 1


# 标签平滑机制
def categorical_crossentropy_with_label_smoothing(y_true, y_pred, label_smoothing=0.1):
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
    return categorical_crossentropy(y_true, y_pred)


if __name__ == '__main__':

    # 模型训练
    train_D = DataGenerator(train_samples)
    dev_D = DataGenerator(dev_samples)
    model = SimpleMultiChoiceMRC(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, MAX_SEQ_LENGTH, NUM_CHOICES).create_model()
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_samples),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        warmup_proportion=WARMUP_RATION,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=2e-5, min_lr=1e-8)
    filepath = "models/multi_choice_model_%s-{epoch:02d}-{val_acc:.4f}.h5" % dataset
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
    model.compile(
        loss=categorical_crossentropy_with_label_smoothing,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=dev_D.__iter__(),
        validation_steps=len(dev_D),
        callbacks=[checkpoint]
    )

    print("finish model training!")

    result = model.evaluate_generator(dev_D.__iter__(), steps=len(dev_D))
    print("model evaluate result: ", result)

    # 模型保存
    # model.save_weights()
    # print("Model saved!")

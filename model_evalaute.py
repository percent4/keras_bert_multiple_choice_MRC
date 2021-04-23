# -*- coding: utf-8 -*-
# @Time : 2021/4/16 12:00
# @Author : Jclian91
# @File : model_evalaute.py
# @Place : Yangpu, Shanghai
import re
import json
import numpy as np
from sklearn.metrics import accuracy_score

from model import SimpleMultiChoiceMRC
from model_train import tokenizer
from params import (dataset,
                    CHECKPOINT_FILE_PATH,
                    CONFIG_FILE_PATH,
                    NUM_CHOICES,
                    MAX_SEQ_LENGTH
                    )

# 加载训练好的模型
model = SimpleMultiChoiceMRC(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, MAX_SEQ_LENGTH, NUM_CHOICES).create_model()
model.load_weights("./models/multi_choice_model_RACE_middle-07-0.6965.h5")

with open("./data/{}/dev.json".format(dataset), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

true_answer_list = []
pred_answer_list = []
for i, sample in enumerate(content):
    article = sample["article"]
    article = re.sub("(（.+?）)", "", article.replace("\u3000", "").replace(" ", "")
                     .replace("\n", "").replace("\r", ""))
    for question, options, true_answer in zip(sample["questions"], sample["options"], sample["answers"]):

        X1 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))
        X2 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))
        question = re.sub("(（.+?）)", "", question.replace("\u3000", "").replace(" ", "")
                         .replace("\n", "").replace("\r", ""))

        for choice_num, option in enumerate(options):
            x1, x2 = tokenizer.encode(first=article, second=question+option, max_len=MAX_SEQ_LENGTH)
            X1[0, choice_num, :] = x1
            X2[0, choice_num, :] = x2

        predict_result = model.predict([X1, X2])
        result = np.argmax(predict_result, axis=1)
        predict_answer = ["A", "B", "C", "D"][result[0]]
        true_answer_list.append(true_answer)
        pred_answer_list.append(predict_answer)
        print(i, true_answer, predict_answer)

        print(f"accuracy: {accuracy_score(true_answer_list, pred_answer_list)}")
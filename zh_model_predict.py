# -*- coding: utf-8 -*-
# @Time : 2021/4/12 14:48
# @Author : Jclian91
# @File : zh_model_predict.py
# @Place : Yangpu, Shanghai
import json
import numpy as np
import pandas as pd

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
model.load_weights("./models/multi_choice_model_haihua_competition-09-0.4632.h5")

with open("./data/{}/validation.json".format(dataset), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

q_ids, q_labels = [], []
for sample in content:
    article = sample["Content"]
    for question_option in sample["Questions"]:
        question = question_option["Question"]
        options = question_option["Choices"]
        q_id = question_option["Q_id"]

        X1 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))
        X2 = np.empty(shape=(1, NUM_CHOICES, MAX_SEQ_LENGTH))

        for choice_num, answer in enumerate(options):
            x1, x2 = tokenizer.encode(first=article, second=question+answer, max_len=MAX_SEQ_LENGTH)
            X1[0, choice_num, :] = x1
            X2[0, choice_num, :] = x2

        predict_result = model.predict([X1, X2])
        result = np.argmax(predict_result)
        predict_answer = ["A", "B", "C", "D"][int(result)]
        q_ids.append(q_id)
        q_labels.append(predict_answer)
        print(q_id, predict_answer)

df = pd.DataFrame({"id": q_ids, "label": q_labels})
df.to_csv("submission_20210420_1709_0.4689.csv", index=False)

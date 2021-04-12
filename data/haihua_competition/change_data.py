# -*- coding: utf-8 -*-
# @Time : 2021/4/12 14:49
# @Author : Jclian91
# @File : change_data.py
# @Place : Yangpu, Shanghai
import json
import uuid
with open("train.json", "r", encoding="utf-8") as f:
    content = json.loads(f.read())

new_samples = []
for sample in content:
    sample_id = sample["ID"]
    sample_article = sample["Content"]
    question_list = []
    options_list = []
    answer_list = []
    for question in sample["Questions"]:
        question_list.append(question["Question"])
        choices = question["Choices"]
        if len(choices) < 4:
            choices += [""] * (4 - len(choices))
        options_list.append(choices)
        answer_list.append(question["Answer"])

    new_samples.append({"id": sample_id,
                        "article": sample_article,
                        "questions": question_list,
                        "options": options_list,
                        "answers": answer_list})

import random
index_range = list(range(len(new_samples)))
random.shuffle(index_range)

train_ration = 0.8
train_samples = [new_samples[i] for i in index_range[:int(0.8*len(index_range))]]
dev_samples = [new_samples[i] for i in index_range[int(0.8*len(index_range)):]]

with open("new_train.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_samples, ensure_ascii=False, indent=4))

with open("new_dev.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(dev_samples, ensure_ascii=False, indent=4))
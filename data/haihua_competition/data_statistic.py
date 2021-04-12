# -*- coding: utf-8 -*-
# @Time : 2021/4/12 15:25
# @Author : Jclian91
# @File : data_statistic.py
# @Place : Yangpu, Shanghai
import re
import json


with open('new_train.json', "r", encoding="utf-8") as f:
    content = json.loads(f.read())

articles = []
for sample in content:
    text = re.sub("(（.+?）)", "", sample["article"].replace("\u3000", "").replace(" ", "").
                  replace("\n", "").replace("\r", "").replace(" ", "").replace("(", "（").replace(")", "）"))
    articles.append(text)

print(sum([len(_) for _ in articles])/len(articles))
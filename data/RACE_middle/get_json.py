# -*- coding: utf-8 -*-
# @Time : 2021/4/10 10:17
# @Author : Jclian91
# @File : get_json.py
# @Place : Yangpu, Shanghai
import os
import json

content = []
for file in os.listdir("."):
    if file.endswith(".txt"):
        with open(file, "r", encoding="utf-8") as f:
            content.append(json.loads(f.read()))

with open("test.json", "w", encoding="utf-8") as g:
    g.write(json.dumps(content, ensure_ascii=False, indent=4))
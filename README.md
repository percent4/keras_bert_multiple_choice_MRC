本项目采用BERT等预训练模型实现多项选择型阅读理解任务（Multiple Choice MRC）

### 维护者

- Jclian91

### 数据集

- RACE Middle: [http://www.cs.cmu.edu/~glai1/data/race/](http://www.cs.cmu.edu/~glai1/data/race/)
- RACE High: [http://www.cs.cmu.edu/~glai1/data/race/](http://www.cs.cmu.edu/~glai1/data/race/)
- haihua competition: [https://www.biendata.net/competition/haihua_2021/data/](https://www.biendata.net/competition/haihua_2021/data/)

### 模型结构

BERT模型

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
token_ids (InputLayer)          (None, 4, 384)       0
__________________________________________________________________________________________________
segment_ids (InputLayer)        (None, 4, 384)       0
__________________________________________________________________________________________________
reshape1 (Lambda)               (None, 384)          0           token_ids[0][0]
__________________________________________________________________________________________________
reshape2 (Lambda)               (None, 384)          0           segment_ids[0][0]
__________________________________________________________________________________________________
model_2 (Model)                 multiple             108891648   reshape1[0][0]
                                                                 reshape2[0][0]
__________________________________________________________________________________________________
cls_layer (Lambda)              (None, 768)          0           model_2[1][0]
__________________________________________________________________________________________________
classifier (Dense)              (None, 1)            769         cls_layer[0][0]
__________________________________________________________________________________________________
reshape3 (Lambda)               (None, 4)            0           classifier[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 4)            0           reshape3[0][0]
==================================================================================================
Total params: 108,892,417
Trainable params: 108,892,417
Non-trainable params: 0
__________________________________________________________________________________________________
```

### 模型效果

- BERT模型

模型参数： NUM_CHOICES = 4, EPOCH = 20, BATCH_SIZE = 5, MAX_SEQ_LENGTH = 360, LEARNING_RATE=2e-5, MIN_LR=1e-8, WARMUP_RATION = 0.1

RACE midele dev数据集上的Accuracy为70.00%.

- Chinese RoBERTa-base模型

模型参数： NUM_CHOICES = 4, EPOCH = 10, BATCH_SIZE = 2, MAX_SEQ_LENGTH = 510, LEARNING_RATE=2e-5, MIN_LR=2e-6, WARMUP_RATION = 0.1

haihua competition 提交结果为46.89%.

### 模型预测

- 初中阅读理解题目

```
{
  "article": "Edward rose early on the New-year morning. He looked in every room and wished a Happy New Year to his family. Then he ran into the street to repeat that to those he might meet.\n\n　　When he came back, his father gave him two bright, new silver dollars.\n\n　　His face lighted up as he took them. He had wished for a long time to buy some pretty books that he had seen at the bookstore.\n\n　　He left the house with a light heart, expecting to buy the books. As he ran down the street, he saw a poor family.\n\n　　“I wish you a Happy New Year.” said Edward, as he was passing on. The man shook his head.\n\n　　“You are not from this country.” said Edward. The man again shook his head, for he could not understand or speak his language. But he pointed to his mouth and to the children shaking with cold, as if (好像) to say, “These little ones have had nothing to eat for a long time.”\n\n　　Edward quickly understood that these poor people were in trouble. He took out his dollars and gave one to the man, and the other to his wife.\n\n　　They were excited and said something in their language, which doubtless meant, “We thank you so much that we will remember you all the time.”\n\n　　When Edward came home, his father asked what books he had bought. He hung his head a moment, but quickly looked up.\n\n　　“I have bought no books”, said he. “I gave my money to some poor people, who seemed to be very hungry then.” He went on, “I think I can wait for my books till next New Year.”\n\n　　“My dear boy,” said his father, “here are some books for you, more as a prize for your goodness of heart than as a New-year gift”\n\n　　“I saw you give the money cheerfully to the poor German family. It was nice for a little boy to do so. Be always ready to help others and every year of your life will be to you a Happy New Year.”",
  "question": "We know that Edward_________ from the passage?",
  "options": [
    "A. got a prize for his kind heart",
    "B. had to buy his books next year",
    "C. bought the books at the bookstore",
    "D. got more money from his father"
  ]
}
```

预测结果为：

```
[[0.32993656 0.5818162  0.07930169 0.00894555]]
正确答案: B
```

### 参考文献

1. RACE_leaderboard: [http://www.qizhexie.com/data/RACE_leaderboard.html](http://www.qizhexie.com/data/RACE_leaderboard.html)
2. BERT for RACE: [https://github.com/NoviScl/BERT-RACE](https://github.com/NoviScl/BERT-RACE)
3. 手把手教你用Pytorch-Transformers——部分源码解读及相关说明（一）: [https://www.cnblogs.com/dogecheng/p/11907036.html](https://www.cnblogs.com/dogecheng/p/11907036.html)
4. [https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py)
5. [https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py)
6. plan to release SWAG code? #38: [https://github.com/google-research/bert/issues/38](https://github.com/google-research/bert/issues/38)
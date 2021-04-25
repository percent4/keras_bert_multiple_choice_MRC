# -*- coding: utf-8 -*-
# @Time : 2021/4/3 17:01
# @Author : Jclian91
# @File : model.py
# @Place : Yangpu, Shanghai
# main architecture of SimpleMultiChoiceMRC
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Activation, MaxPooling1D
from keras_bert import load_trained_model_from_checkpoint


# model structure of SimpleMultiChoiceMRC
class SimpleMultiChoiceMRC(object):
    def __init__(self, config_path, checkpoint_path, max_len, num_choices):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.max_len = max_len
        self.num_choices = num_choices  

    def create_model(self):
        # BERT model
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        for layer in bert_model.layers:
            layer.trainable = True

        # get bert encoder vector
        x1_in = Input(shape=(self.num_choices, self.max_len,), name="token_ids")
        x2_in = Input(shape=(self.num_choices, self.max_len,), name="segment_ids")
        reshape_x1_in = Lambda(lambda x: tf.reshape(x, [-1, self.max_len]), name="reshape1")(x1_in)
        reshape_x2_in = Lambda(lambda x: tf.reshape(x, [-1, self.max_len]), name="reshape2")(x2_in)
        bert_layer = bert_model([reshape_x1_in, reshape_x2_in])
        cls_layer = Lambda(lambda x: x[:, 0], name="cls_layer")(bert_layer)
        logits = Dense(1, name="classifier", activation=None)(cls_layer)
        reshape_layer = Lambda(lambda x: tf.reshape(x, [-1, self.num_choices]), name="reshape3")(logits)
        # log_softmax = Activation(activation=tf.nn.log_softmax)(reshape_layer)
        output = Activation(activation="softmax")(reshape_layer)

        model = Model([x1_in, x2_in], output)
        model.summary()
        # plot_model(model, to_file="model.png")

        return model


if __name__ == '__main__':
    model_config = "./chinese_L-12_H-768_A-12/bert_config.json"
    model_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    model = SimpleMultiChoiceMRC(model_config, model_checkpoint, 400, 4).create_model()
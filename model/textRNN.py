#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : textRNN.py
@Time     : 19-1-8 上午10:45
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import tensorflow.contrib as tc
from model.layers.basic_rnn import rnn


class TextRNN(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by 2 rnn layers and 2 fully connected layers.
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_dim, num_layers, hidden_dim, rnn_type, learning_rate):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.mask_len = tf.placeholder(tf.int32, [None], name='mask_len')
        self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        with tf.name_scope('rnn'):
            # multi-layers rnn network
            _outputs, _ = rnn(rnn_type=rnn_type,
                              inputs=self.embedded_chars,
                              length=self.mask_len,
                              # length=385,
                              hidden_size=hidden_dim,
                              layer_num=num_layers,
                              dropout_keep_prob=self.dropout_keep_prob)
            last = _outputs[:, -1, :]  # batch, time, output_size

        with tf.name_scope('score'):
            # Dense layer, followed a relu activation layer
            fc = tf.layers.dense(last, hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # classifier
            self.logits = tf.layers.dense(fc, num_classes, name='fc2')
            # probability
            self.probs = tf.nn.softmax(self.logits)
            # prediction
            self.y_pred = tf.argmax(self.probs, 1)

        with tf.name_scope('loss'):
            # loss
            self.reg = tc.layers.apply_regularization(tc.layers.l2_regularizer(1e-4), tf.trainable_variables())
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10,
                                                                                labels=self.input_y)
            self.loss = tf.reduce_mean(self.cross_entropy) + self.reg
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.y_pred, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

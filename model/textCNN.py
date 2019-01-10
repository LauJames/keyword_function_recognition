#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : textCNN.py
@Time     : 19-1-8 下午4:10
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tensorflow as tf
import tensorflow.contrib as tc


class TextCNN(object):
    """
    A CNN for text classification
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, learning_rate,
                 embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of L2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_dim, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv"
                    )

                    # Apply no-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool"
                    )
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (un-normalized) scores and predictions
            with tf.name_scope('output'):
                w_output = tf.get_variable(
                    "w_output",
                    shape=[num_filters_total, num_classes],
                    initializer=tc.layers.xavier_initializer()
                )
                b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += tf.nn.l2_loss(w_output)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, w_output, b_output, name='scores')
                self.probs = tf.nn.softmax(self.scores)
                self.y_pred = tf.argmax(self.scores, 1, name='predictions')

            with tf.name_scope('loss'):
                self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(self.cross_entropy) + l2_reg_lambda * l2_loss
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.y_pred, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

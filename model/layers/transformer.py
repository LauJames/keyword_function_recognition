#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Reference: https://github.com/Kyubyong/transformer/blob/master/modules.py
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : transformer.py
@Time     : 2019/1/15 11:10
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tensorflow as tf
import numpy as np


def layer_normalize(inputs,
                    epsilon=1e-8,
                    scope="ln",
                    reuse=None):
    """
    Applies layer normalization.
    :param inputs: A tensor with 2 or more dimensions, where the first dimension has`batch_size`.
    :param epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    :param scope:  Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  # Batch Normalization use 0

        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))

        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta

    return outputs


def postitional_encoding(inputs,
                         num_units,
                         zero_pad=True,
                         scale=True,
                         scope="positional_encoding",
                         reuse=None):
    """
    Positional encoding.
    :param inputs: Tensor. A 2D Tensor with shape of (N, T)
    :param num_units: Int. Decided the output dimensionality.
    :param zero_pad:
    :param scale:
    :param scope:
    :param reuse:
    :return:
    """
    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=True):
        position_index = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        # Github issue has correct the wrong implementation of position encode.
        # position_enc = np.array([[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
        #                           for pos in range(T)])
        position_enc = np.array([[pos / np.power(10000, (i - i % 2)) for i in range(num_units)]
                                 for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, ::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_index)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs


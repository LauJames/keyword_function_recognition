#! /user/bin/evn python
# -*- coding:utf8 -*-

"""
@Reference: https://blog.csdn.net/wxyangid/article/details/79518207
@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : gensim_word2vec.py
@Time     : 19-3-1 下午4:07
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import re
import os
import sys
import logging
import nltk
from gensim.models import word2vec


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)

corpus_path = curdir + '/merged_corpus.txt'
log_path = curdir + '/test.log'
cut_corpus_path = curdir + '/cut_merged_corpus.txt'
vocab_path = curdir + '/vocab.txt'
model_path = curdir + '/train.model'
wv_path = curdir + '/word2vec.txt'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=log_path)


def cut_corpus(corpus, cut_path):
    cut_list = []
    with open(corpus, 'r', encoding='utf8') as fin:
        while True:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                temp = ' '.join(nltk.tokenize(line))
                cut_list.append(temp)

    with open(cut_path, 'w+', encoding='utf8') as fout:
        for temp_line in cut_list:
            fout.write(temp_line + '\n')


def train(cuted_corpus_path, train_model_path, w2v_path, vocab):
    sentences = word2vec.LineSentence(cuted_corpus_path)
    model = word2vec.Word2Vec(sentences, size=12, window=10, min_count=5, worker=5, sg=1, hs=1)
    model.save(train_model_path)
    model.wv.save_word2vec_format(w2v_path, vocab, binary=False)


if __name__ == '__main__':
    train(cut_corpus_path, model_path, wv_path, vocab_path)

    model = word2vec.Word2Vec.load(model_path)
    role1 = ['CNN', 'TCP']
    role2 = ['RNN', 'UDP']
    pairs = [(x, y) for x in role1 for y in role2]

    print(pairs)

    for pair in pairs:
        print("> [%s]和[%s]的相似度为：" % (pair[0], pair[1]), model.similarity(pair[0], pair[1]))  # 预测相似性

    figures = ['SVM', 'Linguistic']
    for figure in figures:
        print("和[%s]最相关的词有: \n" % figure,
              '\n'.join([x[0].ljust(4,'　')+str(x[1]) for x in model.most_similar(figure, topn=10)]),sep='')  # 默认10个最相关  )

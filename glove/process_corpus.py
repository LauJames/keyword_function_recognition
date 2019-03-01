#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : PipelineDemo
@File     : tokenization.py
@Time     : 2019/2/27 22:49
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import sys
import nltk
import spacy
import string
from nltk.corpus import stopwords
import re
from nltk.stem.porter import *

curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)

untoken_corpus_path = curdir + '/csCorpusSampled.txt'

nlp = spacy.load('en_core_web_sm')


def stem_token(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def ie_process(document):
    """Accept string"""
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def get_tokens(document):
    document = document.lower()
    document = ''.join(c for c in document if c not in string.punctuation)
    document = nltk.word_tokenize(document)
    document = [c for c in document if c not in stopwords.words('english')]
    return document


def get_lemma(document):
    document = nlp(document)
    document = ' '.join(token.lemma_ for token in document)
    return document


def noun_chunk(document):
    doc = nlp(document)
    document = [item.text for item in doc.noun_chunks]
    return document


def clean_title(document):
    document = re.split(r'[_-]', document)
    return document


def load_corpus(corpus_path, docs_path):
    corpus_list = []
    with open(corpus_path, 'r', encoding='utf8') as fin:
        while True:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                corpus_list.append(line)
    sentence_list = []
    for doc in corpus_list:
        doc = nlp(doc)
        for sent in doc.sents:
            sentence_list.append(sent)
    del corpus_list
    with open(docs_path, 'w', encoding='utf8') as fout:
        for sent in sentence_list:
            fout.write(sent + '\n')


def split_sent2words(sents_path, split_path):
    docs_split = []
    with open(sents_path, 'r', encoding='utf8') as fin:
        while True:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                line = str(line)
                one_sent = nlp(line)
                sent_blank_split = ' '.join([tmp.text for tmp in one_sent])
                docs_split.append(sent_blank_split)

    with open(split_path, 'w+', encoding='utf8') as fout:
        for doc_split in docs_split:
            fout.write(doc_split + '\n')


if __name__ == '__main__':
    corpus = load_corpus(untoken_corpus_path)


#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : gensim_word2vec.py
@Time     : 19-3-1 下午4:07
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import spacy
import re
import os
import sys
import logging
from gensim.models import word2vec


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)

corpus_path = curdir + '/merged_corpus.txt'
log_path = curdir + '/test.log'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=log_path)

word2vec.Text8Corpus()
#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : vocab.py
@Time     : 19-1-13 下午7:32
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import numpy as np


class Vocab(object):
    """
    Vocabulary preprocessor replacement for tensorflow.contrib.learn.preprocessing.VocabularyProcessor
    """
    def __init__(self, filename=None, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend([self.pad, self.unk_token])
        for token in self.initial_tokens:
            self.add(token)

        if filename is not None:
            self.load_from_file(filename)

    def size(self):
        """
        Get the size of vocabulary.
        :return: Int   An interger indicating the size.
        """
        return len(self.id2token)

    def load_from_file(self, file_path):
        """
        Loads the vocab from file_path
        :param file_path: str   a file with a word in each line.
        :return: None
        """
        for line in open(file_path, 'r'):
            token = line.strip('\n')
            self.add(token)

    def get_id(self, token):
        """
        Gets the id of a token, returns the id of unk token if token id not in vocab.
        :param token: str   a string indicating the word
        :return: Int   An interger
        """
        token = token.lower() if self.lower else token
        try:
            return self.token2id[token]
        except KeyError as e:
            print('Unknow token')
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        Gets the token corresponding to idx, returns unk token id idx is not in vocab
        :param idx: int an interger
        :return: token: str  a token string
        """
        try:
            return self.id2token[idx]
        except KeyError as e:
            print("Unknow index")
            return self.unk_token

    def add(self, token, cnt=1):
        """
        Adds the token to vocab.
        :param token: str
        :param cnt: int  a num indicating the count of the token to add, default is 1
        :return: idx: int
        """
        token = token.lower() if self.lower else token
        if token in self.token2id:
            # Token exist. Gets the idx straightforwardly.
            idx = self.token2id[token]
        else:
            # Token doesn't exist. Push into the id2token and token2id dict.
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        Filter the tokens in vocab by their count.
        :param min_cnt: int   tokens with frequency less than min_cnt is filtered
        :return: None
        """
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def randomly_init_embeddings(self, embed_dim):
        """
        Randomly initializes the embeddings for each token.
        :param embed_dim: int   the size of the embedding for each token.
        :return: None
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros([self.embed_dim])

    def load_pretrained_embeddings(self, embedding_path):
        """
        Load the pretrained word embeddings from embedding_path.
        Reconstructed the token2id and id2token dict. Tokens not in pretrained embeddings will be filtered.
        :param embedding_path: str
        :return: None
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            while True:
                line = fin.readline()
                if not line:
                    print("Pretrained embeddings load successfully!")
                    break
                contents = line.strip().split(' ')
                token = contents[0]
                if token not in self.token2id:
                    continue
                trained_embeddings[token] = list(map(float, contents[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        # load embeddings
        self.embeddings = np.zeros([self.size(), self.embed_dim])
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def convert2ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        :param tokens: list   A list of tokens
        :return: list   A list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered.
        :param ids: list   A list of ids to convert.
        :param stop_id:  int   The stop id, default is None.
        :return:
        """
        tokens = []
        for i in ids:
            tokens.append(self.get_token(i))
            if stop_id is not None and i == stop_id:
                break
        return tokens

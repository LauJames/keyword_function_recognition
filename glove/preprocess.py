#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : PipelineDemo
@File     : preprocess.py
@Time     : 2019/2/27 22:49
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import os
import sys
import re
import random


# random.seed(7)


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, os.path.dirname(curdir))

original_text_path = curdir + '/csV1.txt'  # 289933 papers
temp_save_path = curdir + '/csCorpus.txt'
sampled_corpus_path = curdir + '/csCorpusSampled.txt'


def sampling(corpus_path, sample_corpus_path):
    sampled_list = []
    total_count = 0
    sample_count = 0
    with open(corpus_path, 'r', encoding='utf8') as fin:
        while True:
            lines = fin.readlines(100000)
            if not lines:
                break
            for line in lines:
                total_count += 1
                random_num = random.randrange(1, 5, 1)
                if random_num == 7:
                    temp = line.strip()
                    sampled_list.append(temp)
                    sample_count += 1
        print('total lines count:' + str(total_count))
        print('sampling count:' + str(sample_count))

    with open(sample_corpus_path, 'w+', encoding='utf8') as fout:
        for sample in sampled_list:
            fout.write(sample + '\n')
        print('Sampled corpus write to:' + sample_corpus_path)


def clean_str(string):
    # string = re.sub(r'===============================================.*$', '', string, 1)
    # string = re.sub(r'===============================================', ' ', string)
    string = re.sub(r'<a.+?/a>', '', string)
    string = re.sub(r'\{math_begin\}.+?\{math_end\}', 'MATHFORMULA', string)
    string = re.sub(r'\s+', ' ', string)
    temp_list = string.split('===============================================')
    if len(temp_list) == 3:
        string = temp_list[0] + temp_list[1]
        return string
    else:
        return None

def get_paper(path, save_path):
    paper_list = []
    with open(path, 'r', encoding='utf8') as fin:
        one_paper = ' '
        while True:
            lines = fin.readlines(100000)
            if not lines:
                break

            for line in lines:
                if line.strip().replace('\n', '') == '########################################++++++++++++++++++++++++++++###################################':
                    one_paper = clean_str(one_paper)
                    if one_paper is None:
                        one_paper = ' '
                        continue
                    paper_list.append(one_paper)
                    one_paper = ' '
                else:
                    temp = line.strip().replace('\n', '')
                    if temp is not None:
                        one_paper = one_paper + ' ' + temp

    with open(save_path, 'w+', encoding='utf8') as fout:
        for paper in paper_list:
            fout.write(paper + '\n')


if __name__ == '__main__':
    get_paper(original_text_path, temp_save_path)
    # sampling(temp_save_path, sampled_corpus_path)

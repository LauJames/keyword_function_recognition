#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : keyword_function_recognition 
@File     : key_phrase_extract.py
@Time     : 2019/3/4 10:46
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from rake_nltk import Rake

r = Rake()


if __name__ == '__main__':
    my_test = 'My father was a self-taught mandolin player. He was one of the best string instrument players ' \
              'in our town. He could not read music, but if he heard a tune a few times, he could play it. ' \
              'When he was younger, he was a member of a small country music band. They would play at local dances ' \
              'and on a few occasions would play for the local radio station. He often told us how he had auditioned ' \
              'and earned a position in a band that featured Patsy Cline as their lead singer. He told the family that ' \
              'after he was hired he never went back. Dad was a very religious man. He stated that there was a lot of ' \
              'drinking and cursing the day of his audition and he did not want to be around that type of environment.'
    r.extract_keywords_from_text(my_test)
    print(r.get_ranked_phrases())
    print(r.get_ranked_phrases_with_scores())
    print(r.get_word_degrees())

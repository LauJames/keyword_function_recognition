#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : science_vec
@File     : preprocess.py
@Time     : 19-1-4 下午3:30
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import os
import sys
import MySQLdb


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)


save_path = curdir + '/acm_summary_title_corpus.txt'


def get_summary_from_mysql():
	conn = MySQLdb.connect(user='root', password='irlab2018', charset='utf8', database='ACM')
	cursor = conn.cursor()
	summary_list = []
	title_list = []
	try:
		# total lines: 295567
		sql_str = 'select summary, title from acm_paper;'
		cursor.execute(sql_str)
		# split pages
		limits = 1000
		for page in range(297):
			values = cursor.fetchmany(limits)
			for temp in values:
				summary_list.append(temp[0])
				title_list.append(temp[1])

	except MySQLdb.Error as e:
		print(e)

	finally:
		cursor.close()
		conn.close()

	return summary_list, title_list


def save_title_summary(summary_list, title_list, out_path):
	with open(out_path, 'w+', encoding='utf8') as fout:
		for summary, title in zip(summary_list, title_list):
			summary = summary.replace('\n', '')
			title = title.replace('\n', '')
			if summary.strip() == 'An abstract is not available.':
				summary = ''
			fout.write(title + ' ' + summary + '\n')


if __name__ == '__main__':
	summaries, titles = get_summary_from_mysql()
	save_title_summary(summaries, titles, save_path)


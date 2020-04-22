#!/usr/bin/env python
# encoding=utf-8
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

from __future__ import print_function
import sys
import jieba
reload(sys)
sys.setdefaultencoding('utf-8')

trans_file=sys.argv[1]

for word in open('local/jieba_words.txt'):
    jieba.suggest_freq(word.strip("\n"), tune=True)

for line in open(trans_file):
  key,trans = line.strip().split(' ',1)
  words = jieba.cut(trans.replace(" ",""), HMM=False) # turn off new word discovery (HMM-based)
  new_line = key + '\t' + " ".join(words)
  print(new_line)

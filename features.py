#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import random
import gensim
import numpy as np
import math
import logging

from radical import Radical
import pinyin
import csv

def calEditDistance(m,n):
    """compute the least steps number to convert m to n by insert , delete , replace .
    动态规划算法,计算单词距离
    print word_distance("abc","abec")
    1
    print word_distance("ababec","abc")
    3
    """
    len_1=lambda x:len(x)+1
    c=[[i] for i in range(0,len_1(m)) ]
    c[0]=[j for j in range(0,len_1(n))]

    for i in range(0,len(m)):
    #    print i,' ',
        for j in range(0,len(n)):
            c[i+1].append(
                min(
                    c[i][j+1]+1,#插入n[j]
                    c[i+1][j]+1,#删除m[j]
                    c[i][j] + (0 if m[i]==n[j] else 1 )#改
                )
            )
    #        print c[i+1][j+1],m[i],n[j],' ',
    #    print ''
    edit_distance = c[-1][-1]
    edit_distance_score = 1.0 - (edit_distance+0.0) / max(len(m),len(n))
    return edit_distance_score# c[-1][-1]


def calPinyinEditDistance(word1,word2):
    # p = pinyin()
    # word1_pinyin = p.get_pinyin(word1)#.decode())
    # word2_pinyin = p.get_pinyin(word2)#.decode())
    #
    # word1_pinyin = word1_pinyin.replace('-','')
    # word2_pinyin = word2_pinyin.replace('-','')

    word1_pinyin = pinyin.get(word1)
    word2_pinyin = pinyin.get(word2)

    return calEditDistance(word1_pinyin,word2_pinyin)


def calCommonRadical(word1,word2,radical):

    # word1 = word1.decode('utf-8')
    # word2 = word2.decode('utf-8')

    dict1 = {}
    for alph in word1:
        radical_ = radical.get_radical(alph)

        if radical_ not in dict1:
            dict1[radical_] = 1
        else:
            dict1[radical_] += 1

    count = 0
    for alph in word2:
        radical_ = radical.get_radical(alph)
        if radical_ in dict1 and dict1[radical_]>0:
            count += 1
            dict1[radical_] -= 1

    return float(count)/ max(len(word1),len(word2))



#从csv文件中读取baidu相似度的结果，然后存到dict中
def getSearchScoreFromCSV(filename):
    dic = dict()

    f = open(filename,'r')
    for line in f:
    # reader = csv.reader(file(filename, 'rb'))
    # for line in reader:
    #     # print line[0],line[1]#.strip('|')
    #     print line
        line = line.split('|')
        tmp = line[2:]
        dic[(line[0],line[1])] = tmp#dic.get(line[0],[]) + tmp#line[1].split('|')#.remove('')



    return dic

def calSearchScore(dic,word1,word2):
    try:
        tmp = dic[(word1,word2)]
    except:
        print word1,word2
        return 0
    # print tmp
    fx = int(tmp[0])
    fy = int(tmp[1])
    if tmp[2] == '':
        fxy = 0
    else:
        fxy = int(tmp[2])

    fx += 1
    fy += 1
    fxy += 1

    M = math.log(100000000*1000,2)

    nominator = max(math.log(fx,2),math.log(fy,2)) - math.log(fxy,2)
    denominator = M - min(math.log(fx,2),math.log(fy,2))
    # nominator = max(math.log10(fx),math.log10(fy)) - math.log10(fxy)
    # denominator = M - min(math.log10(fx),math.log10(fy))
    # print fx,fy,fxy
    # print nominator,denominator,1 - nominator/denominator
    return 1 - nominator/denominator









# print calPinyinEditDistance('你们好','你们')
#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

import gensim
import logging

import features
import radical
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# from gensim.core import multiarray
# import translator

try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 对象是语料库，一部分预处理
def analyze_txt():

    # f = open('data/seged text/text_EMR_seged.txt','r')
    f = open('data/seged text/wiki_ALL_seged.txt','r')
    count = 0
    count_of_words = 0
    res = ''
    for line in f:
        count += 1
        res = line
        count_of_words += len(line)
    print count
    print count_of_words
    print line

    # print 2020609807 * 3 / 1000000000

    return

# 删除停用词和OOV
def get_words_without_dups():

    def write_file(path,content):
        with open(path, mode='a') as f:
            for i in content:
                f.writelines(i + '\n')
            f.close()
    res = set()
    f = open('data/syns/syns_ALL_1214.txt','r')
    count = 0
    for line in f:
        line = line.strip()
        print(line)
        res.add(line)

    res = list(res)
    write_file('data/syns/syns_ALL_1214_without_dups.txt',res)

def get_words_not_OOV(model_name = 'model_DB_neike_abs_RE_wiki',filename = 'data/syns/syns_ALL_1214_without_dups.txt'):

    model = gensim.models.Word2Vec.load('model/%s' % (model_name))
    dic = model.vocab

    # f = open('data/syns/syns_ALL_1214.txt','r')
    f = open(filename,'r')
    count = 0
    total = 0
    res = []
    for line in f:
        total += 1
        line = line.strip()
        # print(line)
        line = line.split('|')
        e1 = line[0]
        e2 = line[1]

        if e1 not in dic or e2 not in dic:
            count += 1
            # print e1,e2
            continue
        res += (e1,e2),
    print count,total
    return res

def write_file(path,content):
    with open(path, mode='a') as f:
        for i in content:
            f.writelines(i + '\n')
        f.close()

#下面这一部分都是用于获得model

def loadFileToTrain(filepath):
    f = open(filepath,'r')
    count = 0
    sentences = []

    for line in f:
        count += 1
        if count % 10000 == 0:
            print count
        # if count < start:
        #     continue
        # if count >= 50000000:
        # if count >= start + num_of_sentences:
        #     break
        sentence = []
        line = line.replace('　','')#.decode('GB2312','ignore').encode('utf-8')
        temp = line.split(' ')
        for i in temp:
            if i == ' ' or i == '' or i == ' ':
                continue
            #print i
            #print chardet.detect(i)
            i.strip()
            i.lstrip()
            i.rstrip()
            if len(i) > 2:
                sentence.append(i)
        if len(sentence) > 2:
            sentences.append(sentence)

    return sentences

#根据文件内容进行训练
def Load_file_and_train(filepath, model_name):


    sentences = loadFileToTrain(filepath)

    model = gensim.models.Word2Vec(sentences, workers = 4)

    # model.save('model/word2vec_model_wiki_half_with_radical')

    model.save('model\\' + model_name)

    print model.vocab.keys()[0]
    # print model['土']

#只提供句子，进行训练
def Load_sentences_and_train(sentences,model_name,window_size=5):


    model = gensim.models.Word2Vec(sentences, min_count = 5,workers = 4)

    # model.save('model/word2vec_model_wiki_half_with_radical')

    model.save('model\\' + model_name)

    print model.vocab.keys()[0]
    # print model['土']

# 读取多个文件然后训练
def load_files_and_train(model_name):

    sents = loadFileToTrain('data/seged text/wiki_ALL_seged.txt')
    print len(sents)
    sents += loadFileToTrain('data/seged text/DB_ALL_seged.txt')
    print len(sents)
    sents += loadFileToTrain('data/seged text/neike_ALL_seged.txt')
    print len(sents)
    sents += loadFileToTrain('data/seged text/baike_ALL_seged.txt')
    print len(sents)
    sents += loadFileToTrain('data/seged text/RE_ALL_seged.txt')
    print len(sents)

    # sents = []
    # for i in files:
    #     sents += load_files_and_train(i)
    #     print len(sents)


    model = gensim.models.Word2Vec(sents, workers = 4)

    # model.save('model/word2vec_model_wiki_half_with_radical')

    model.save('model\\' + model_name)

    print model.vocab.keys()[0]
    print len(model.vocab.keys())
    # print model['土']

def incremental_train(filepath,model_name):
    sentences = loadFileToTrain(filepath,'',0,1000000000000)

    print len(sentences)

    model = gensim.models.Word2Vec.load('model/%s' % (model_name))
    model.train(sentences, total_examples=model.corpus_count)

    model.save('model\\' + 'test')

    print model.vocab.keys()[0]
    print len(model.vocab)

# 从文件中读取word2vec向量
def get_word2vec_from_file(path):
    f = open(path,'r')
    res = []
    words1 =[]
    words2 =[]
    vec1, vec2 = [],[]
    for line in f:
        line = line.strip()
        line = line.split('|||')
        p1 = line[0].split('|')
        p2 = line[1].split('|')

        words1 += p1[0],
        words2 += p2[0],

        vec1 += [float(i) for i in p1[1].split(',')],
        vec2 += [float(i) for i in p2[1].split(',')],

        # res += float(line),
    return words1,words2,vec1,vec2

# 把根据word2vec计算出来的相似度作为baseline写进文件里，文件是一维向量
def write_word2vec_similarity_to_file():

    # 正样本的
    # res_all = get_words_not_OOV()

    #负样本的 医学
    # res_all = get_words_not_OOV(filename='data/neg_syns/med_negative_100w_De_OOV.txt')

    # 负样本的 不止包含医学的
    res_all = get_words_not_OOV(filename='data/neg_syns/word_from_model_negative_100w.txt')
    res_all = res_all[:50000]

    data_all = cal_similarity(res_all)

    # print data_all

    content = []

    for i in data_all:
        content += str(i),

    write_file('data/data_vector/not_only_med_50000_similarity_score_baseline_negative.txt',content)

# 直接把word2vec的结果写进文件里，文件是100维的
def write_word2vec_to_file():

    # 正样本的
    res_all = get_words_not_OOV()

    #负样本的 医学
    # res_all = get_words_not_OOV(filename='data/neg_syns/med_negative_100w_De_OOV.txt')

    # 负样本的 不止包含医学的
    # res_all = get_words_not_OOV(filename='data/neg_syns/word_from_model_negative_100w.txt')


    res_all = res_all[:50000]

    model_name = 'model_DB_neike_abs_RE_wiki'#,filename = 'data/syns/syns_ALL_1214_without_dups.txt'):
    model = gensim.models.Word2Vec.load('model/%s' % (model_name))
    # dic = model.vocab

    content = []

    for i,j in res_all:
        v1 = list(model[i])
        v1 = ['{:.15f}'.format(x) for x in v1]

        v2 = list(model[j])
        v2 = ['{:.15f}'.format(x) for x in v2]
        # print v1
        content += i + '|' + ','.join(v1) + '|||' + j + '|' + ','.join(v2),

    write_file('data/data_vector/word2vec_med_7666_positive.txt',content)
    # write_file('data/data_vector/word2vec_med_50000_negative.txt',content)
    # write_file('data/data_vector/word2vec_not_only_med_50000_negative.txt',content)

# 从文件中读取相似度的score，这个是作为baseline的
def get_similarity_score_from_file(path):
    f = open(path,'r')
    res = []
    for line in f:
        res += float(line),
    return res

def cal_similarity(s, model_name = 'model_DB_neike_abs_RE_wiki'):

    model = gensim.models.Word2Vec.load('model/%s' % (model_name))
    res = []
    for (i,j) in s:
        # print i,j,model.similarity(i,j)
        res += model.similarity(i,j),
    res.sort()
    print res[-100:]
    return res

def deal_with_data_for_curve(res,inter = 0.03):
    start = -1#0.5#res[0]
    end = 1.0#res[-1]
    print start,end

    data = [0] * int((end - start)/inter + 1)
    for i in res:
        tmp = int((i - start) / inter)
        data[tmp] += 1
    return data,start,end,inter

def deal_with_data_for_curve_and_get_percents(res,inter = 0.03):
    start = -0.5#res[0]
    end = 1.0#res[-1]
    print start,end

    data = [0] * int((end - start)/inter + 1)
    for i in res:
        tmp = int((i - start) / inter)
        data[tmp] += 1
    total = len(res)
    for i in range(len(data)):
        data[i] = data[i] /(total + 0.0)
    return data,start,end,inter


def draw_curve(res,start,end,inter,name):
    from numpy import *
    import matplotlib.pyplot as plt
    #画曲线图
    fig = plt.figure()
    y = []
    while start <= end:
        start += inter
        y += start,
    plt.plot(y,res, 'r', linewidth=2)
    plt.xlabel(r'x', fontsize=16)
    plt.ylabel(r'similarity', fontsize=16)
    plt.title(r'', fontsize=16)
    plt.show()
    plt.savefig(name)



def cal_and_draw_curve(inter):
    # 这个是开始测试结果的
    model_name_all = 'model_DB_neike_abs_RE_wiki'
    res_all = get_words_not_OOV()
    data_all = cal_similarity(res_all)
    data_all,start_all,end_all,inter_all = deal_with_data_for_curve(data_all, inter)
    # draw_curve(data_all,start,end,inter,model_name)


    model_name_db = 'model_DB'
    res_db = get_words_not_OOV(model_name_db)
    data_db = cal_similarity(res_db, model_name_db)
    data_db,start_db,end_db,inter_db = deal_with_data_for_curve(data_db, inter)


    model_name_emr = 'model_EMR'
    res_emr = get_words_not_OOV(model_name_emr)
    data_emr = cal_similarity(res_emr, model_name_emr)
    data_emr,start_emr,end_emr,inter_emr = deal_with_data_for_curve(data_emr, inter)

    model_name_re = 'model_RE'
    res_re = get_words_not_OOV(model_name_re)
    data_re = cal_similarity(res_re, model_name_re)
    data_re,start_re,end_re,inter_re = deal_with_data_for_curve(data_re, inter)


    model_name_baike = 'model_Baike'
    res_baike = get_words_not_OOV(model_name_baike)
    data_baike = cal_similarity(res_baike, model_name_baike)
    data_baike,start_baike,end_baike,inter_baike = deal_with_data_for_curve(data_baike, inter)



    model_name_neike = 'model_Neike'
    res_neike = get_words_not_OOV(model_name_neike)
    data_neike = cal_similarity(res_neike, model_name_neike)
    data_neike,start_neike,end_neike,inter_neike = deal_with_data_for_curve(data_neike, inter)
    # draw_curve(data,start,end,inter,model_name)


    start = min(start_all,start_db)
    end = max(end_all,end_db)

    import matplotlib.pyplot as plt
    #画曲线图
    fig = plt.figure()
    y = [0] * len(data_all)
    i = 0
    while i < len(data_all):
        y[i] = start + inter * i
        i += 1

    print y
    print len(y),len(data_all),len(data_db)
    plt.plot(y,data_all, 'r', linewidth=2,label='data_all')
    plt.plot(y,data_db, 'b', linewidth=2,label='data_db')
    plt.plot(y,data_emr, 'darkorange', linewidth=2,label='data_emr')
    plt.plot(y,data_re, 'darkblue', linewidth=2,label='data_re')
    plt.plot(y,data_baike, 'g', linewidth=2,label='data_baike')
    plt.plot(y,data_neike, 'c', linewidth=2,label='data_neike')
    plt.xlabel(r'similarity', fontsize=16)
    plt.ylabel(r'number', fontsize=16)
    plt.title(r'Similarity', fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig(('compare_%s' % (inter)).replace('.',''))
    plt.show()


def cal_and_draw_curve_and_get_percent(inter):
    # 这个是开始测试结果的
    model_name_all = 'model_DB_neike_abs_RE_wiki'
    res_all = get_words_not_OOV()
    data_all = cal_similarity(res_all)
    data_all,start_all,end_all,inter_all = deal_with_data_for_curve_and_get_percents(data_all, inter)
    # draw_curve(data_all,start,end,inter,model_name)


    model_name_db = 'model_DB'
    res_db = get_words_not_OOV(model_name_db)
    data_db = cal_similarity(res_db, model_name_db)
    data_db,start_db,end_db,inter_db = deal_with_data_for_curve_and_get_percents(data_db, inter)


    model_name_emr = 'model_EMR'
    res_emr = get_words_not_OOV(model_name_emr)
    data_emr = cal_similarity(res_emr, model_name_emr)
    data_emr,start_emr,end_emr,inter_emr = deal_with_data_for_curve_and_get_percents(data_emr, inter)
    # draw_curve(data,start,end,inter,model_name)

    model_name_re = 'model_RE'
    res_re = get_words_not_OOV(model_name_re)
    data_re = cal_similarity(res_re, model_name_re)
    data_re,start_re,end_re,inter_re = deal_with_data_for_curve_and_get_percents(data_re, inter)


    model_name_baike = 'model_Baike'
    res_baike = get_words_not_OOV(model_name_baike)
    data_baike = cal_similarity(res_baike, model_name_baike)
    data_baike,start_baike,end_baike,inter_baike = deal_with_data_for_curve_and_get_percents(data_baike, inter)


    model_name_neike = 'model_Neike'
    res_neike = get_words_not_OOV(model_name_neike)
    data_neike = cal_similarity(res_neike, model_name_neike)
    data_neike,start_neike,end_neike,inter_neike = deal_with_data_for_curve_and_get_percents(data_neike, inter)
    # draw_curve(data,start,end,inter,model_name)


    start = min(start_all,start_db)
    end = max(end_all,end_db)

    import matplotlib.pyplot as plt
    #画曲线图
    fig = plt.figure()
    y = [0] * len(data_all)
    i = 0
    while i < len(data_all):
        y[i] = start + inter * i
        i += 1
    print y
    print len(y),len(data_all),len(data_db)
    plt.plot(y,data_all, 'r', linewidth=2,label='data_all')
    plt.plot(y,data_db, 'b', linewidth=2,label='data_db')
    plt.plot(y,data_emr, 'darkorange', linewidth=2,label='data_emr')
    plt.plot(y,data_re, 'darkblue', linewidth=2,label='data_re')
    plt.plot(y,data_baike, 'g', linewidth=2,label='data_baike')
    plt.plot(y,data_neike, 'c', linewidth=2,label='data_neike')
    plt.xlabel(r'similarity', fontsize=16)
    plt.ylabel(r'percentage', fontsize=16)
    plt.title(r'Similarity', fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.savefig(('compare_percentage_%s' % (inter)).replace('.',''))
    plt.show()


def get_neg_data_from_model(model_name = 'model_DB_neike_abs_RE_wiki'):

    model = gensim.models.Word2Vec.load('model/%s' % (model_name))

    res = []

    dic = model.vocab.keys()
    for i in dic:
        flag = 0
        for ii in '0987654321':
            if ii in i:
                flag = 1
                break
        for ii in 'qwertyuioplkjhgfdsazxcvbnmQWERTYUIOPLKJHGFDSAZXCVBNM':
            if ii in i:
                flag = 1
                break
        if flag == 0:
            res += i.strip(),
    # total = len(dic)
    # print dic
    write_file('data/neg_syns/vocab_from_model.txt',res)

def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)




def cal_similarity_with_other_features(w1,w2,v1,v2, other_features, model_name = 'model_DB_neike_abs_RE_wiki'):

    # model = gensim.models.Word2Vec.load('model/%s' % (model_name))
    res = []
    for i in range(len(w1)):
        # print i,j,model.similarity(i,j)
        # e1 = s[i][0]
        # e2 = s[i][1]
        v1[i] += [1] * 10 *(len(other_features[i]))
        v2[i] += other_features[i] * 10
        # print v1[i],type(v1[i])
        # print v2[i],type(v2[i])
        res += cos(v1[i],v2[i]),
        # print len(v1),len(v2),cos(v2,v2)
        # return
    res.sort()
    # print res[-100:]
    return res

# def cal_similarity(s, model_name = 'model_DB_neike_abs_RE_wiki'):
#
#     model = gensim.models.Word2Vec.load('model/%s' % (model_name))
#     res = []
#     for (i,j) in s:
#         # print i,j,model.similarity(i,j)
#         res += model.similarity(i,j),
#     res.sort()
#     print res[-100:]
#     return res



# # 专门给医学同义词测试并画曲线用的
# def cal_and_draw_curve_with_other_features_for_medical(inter = 0.03):
#
#     inter = 0.03
#
#
#     model_name_all = 'model_DB_neike_abs_RE_wiki'
#     res_all = get_words_not_OOV()
#     data_all = cal_similarity(res_all)
#     data_all,start_all,end_all,inter_all = deal_with_data_for_curve(data_all, inter)
#     # draw_curve(data_all,start,end,inter,model_name)
#
#     baidu_dic = features.getSearchScoreFromCSV('data/baidu_7666.csv')
#     # for k,v in baidu_dic.items():
#     #     print k,v
#
#     other_features = []
#
#     radical_ = radical.Radical()
#     baidu_tmp = []
#     for i in res_all:
#         baidu_tmp += features.calSearchScore(baidu_dic,i[0],i[1]),
#         other_features += [
#                               deal_with_feature_linear(features.calEditDistance(i[0],i[1])),
#                            deal_with_feature_linear(features.calPinyinEditDistance(i[0],i[1])),
#                            deal_with_feature_linear(features.calCommonRadical(i[0],i[1],radical_)),
#                            # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
#                            # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
#                            # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
#                            # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1]))
#                            deal_with_feature_linear(features.calSearchScore(baidu_dic,i[0],i[1]),0.4),
#                            deal_with_feature_linear(features.calSearchScore(baidu_dic,i[0],i[1]),0.4),
#                            # deal_with_feature_linear(features.calSearchScore(baidu_dic,i[0],i[1]),0.6),
#                            deal_with_feature_linear(features.calSearchScore(baidu_dic,i[0],i[1]),0.4)
#                           ],
#
#
#     for i in range(len(res_all)):
#         print res_all[i][0],res_all[i][1],other_features[i]
#
#     # print baidu_tmp
#         # other_features += [features.calEditDistance(i[0],i[1]),features.calPinyinEditDistance(i[0],i[1]),features.calCommonRadical(i[0],i[1],radical_)],
#         # other_features += [features.calCommonRadical(i[0],i[1],radical)],
#     # for i in range(len(res_all)):
#     #     print res_all[i][0],res_all[i][1],other_features[i]
#
#     # print other_features
#     data_all_other = cal_similarity_with_other_features(res_all,other_features)
#     data_all_other,start_all_other,end_all_other,inter_all_other = deal_with_data_for_curve(data_all_other, inter)
#



def cal_and_add_features(w1,w2, baidu_dic):

    other_features = []


    radical_ = radical.Radical()
    # baidu_tmp = []
    for i in range(len(w1)):
        # baidu_tmp += features.calSearchScore(baidu_dic,i[0],i[1]),
        other_features += [
                              deal_with_feature_linear(features.calEditDistance(w1[i],w2[i])),
                           deal_with_feature_linear(features.calPinyinEditDistance(w1[i],w2[i])),
                           deal_with_feature_linear(features.calCommonRadical(w1[i],w2[i],radical_)),
                           # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
                           # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
                           # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1])),
                           # deal_with_baidu_linear(features.calSearchScore(baidu_dic,i[0],i[1]))
                           deal_with_feature_linear(features.calSearchScore(baidu_dic,w1[i],w2[i]),0.4),
                           deal_with_feature_linear(features.calSearchScore(baidu_dic,w1[i],w2[i]),0.4),
                           # deal_with_feature_linear(features.calSearchScore(baidu_dic,i[0],i[1]),0.6),
                           deal_with_feature_linear(features.calSearchScore(baidu_dic,w1[i],w2[i]),0.4)
                          ],

    return other_features

# 获取data数据结果的，为了计算相关系数
# 正样本
def cal_and_get_data_res_for_medical(inter=0.03, feature_num=6):
    data_baseline = get_similarity_score_from_file(
        'data/baseline score/med_7666_similarity_score_baseline_positive.txt')
    data_baseline, start_all, end_all, inter_all = deal_with_data_for_curve(data_baseline, inter)
    # draw_curve(data_all,start,end,inter,model_name)

    baidu_dic = features.getSearchScoreFromCSV('data/baidu_7666.csv')

    words1, words2, w2v1, w2v2 = get_word2vec_from_file('data/data_vector/word2vec_med_7666_positive.txt')
    other_features = cal_and_add_features(words1, words2, baidu_dic)

    other_features = [i[:feature_num] for i in other_features]

    # for i in other_features[:5]:
    #     print(i)

    data_with_features = cal_similarity_with_other_features(words1, words2, w2v1, w2v2, other_features)

    res = []

    for i in range(len(w2v1)):
        res += cos(w2v1[i], w2v2[i]),
    return res

# 计算负样本的data结果，为了计算相关系数的
def cal_and_get_data_res_for_medical_negative(inter = 0.03,if_all_model = True, feature_num =6):

    if if_all_model:
        data_baseline = get_similarity_score_from_file('data/baseline score/not_only_med_50000_similarity_score_baseline_negative.txt')

    else:
        data_baseline = get_similarity_score_from_file('data/baseline score/med_50000_similarity_score_baseline_negative.txt')
    # data_baseline,start_all,end_all,inter_all = deal_with_data_for_curve(data_baseline, inter)

    if if_all_model:
        baidu_dic = features.getSearchScoreFromCSV('data/baidu_nagetive_all_model_vocab_1221.csv')
    else:
        baidu_dic = features.getSearchScoreFromCSV('data/baidu_nagetive_medical_1221.csv')

    if if_all_model:
        words1,words2,w2v1,w2v2 = get_word2vec_from_file('data/data_vector/word2vec_not_only_med_50000_negative.txt')
    else:
        words1,words2,w2v1,w2v2 = get_word2vec_from_file('data/data_vector/word2vec_med_50000_negative.txt')

    other_features = cal_and_add_features(words1,words2, baidu_dic)

    other_features = [i[:feature_num] for i in other_features]

    # for i in other_features[:5]:
    #     print(i)

    data_with_features = cal_similarity_with_other_features(words1,words2,w2v1,w2v2,other_features)
    # data_with_features,start_all_other,end_all_other,inter_all_other = deal_with_data_for_curve(data_with_features, inter)


    res = []

    for i in range(len(w2v1)):
        res += cos(w2v1[i],w2v2[i]),
    res.sort()
    return res
# 专门给医学同义词测试并画曲线用的
# 正样本的
def cal_and_draw_curve_with_other_features_for_medical(inter = 0.03):

    data_baseline = get_similarity_score_from_file('data/baseline score/med_7666_similarity_score_baseline_positive.txt')
    data_baseline,start_all,end_all,inter_all = deal_with_data_for_curve(data_baseline, inter)
    # draw_curve(data_all,start,end,inter,model_name)

    baidu_dic = features.getSearchScoreFromCSV('data/baidu_7666.csv')


    words1,words2,w2v1,w2v2 = get_word2vec_from_file('data/data_vector/word2vec_med_7666_positive.txt')
    other_features = cal_and_add_features(words1,words2, baidu_dic)

    data_with_features = cal_similarity_with_other_features(words1,words2,w2v1,w2v2,other_features)
    data_with_features,start_all_other,end_all_other,inter_all_other = deal_with_data_for_curve(data_with_features, inter)



    start = -1.0#min(start_all,start_db)
    end = 1.0#max(end_all,end_db)

    import matplotlib.pyplot as plt
    #画曲线图
    fig = plt.figure()
    y = [0] * len(data_baseline)
    i = 0
    while i < len(data_baseline):
        y[i] = start + inter * i
        i += 1
    plt.plot(y,data_baseline, 'r', linewidth=2,label='data_all')
    # data_all = [0,0,0,0,0,0,0] + data_all[:-7]
    # plt.plot(y,data_all, 'b', linewidth=2,label='data_all')
    plt.plot(y,data_with_features, 'b', linewidth=2,label='data_all_other')
    plt.xlabel(r'similarity', fontsize=16)
    plt.ylabel(r'number', fontsize=16)
    plt.title(r'', fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig(('compare_with_other_features_%s' % (inter)).replace('.',''))
    plt.show()

# 专门给医学词测试并画曲线用的
# 负样本的
def cal_and_draw_curve_with_other_features_for_medical_negative(inter = 0.03,if_all_model = True):
    if if_all_model:
        data_baseline = get_similarity_score_from_file('data/baseline score/not_only_med_50000_similarity_score_baseline_negative.txt')

    else:
        data_baseline = get_similarity_score_from_file('data/baseline score/med_50000_similarity_score_baseline_negative.txt')
    data_baseline,start_all,end_all,inter_all = deal_with_data_for_curve(data_baseline, inter)

    if if_all_model:
        baidu_dic = features.getSearchScoreFromCSV('data/baidu_nagetive_all_model_vocab_1221.csv')
    else:
        baidu_dic = features.getSearchScoreFromCSV('data/baidu_nagetive_medical_1221.csv')

    if if_all_model:
        words1,words2,w2v1,w2v2 = get_word2vec_from_file('data/data_vector/word2vec_not_only_med_50000_negative.txt')
    else:
        words1,words2,w2v1,w2v2 = get_word2vec_from_file('data/data_vector/word2vec_med_50000_negative.txt')

    other_features = cal_and_add_features(words1,words2, baidu_dic)

    data_with_features = cal_similarity_with_other_features(words1,words2,w2v1,w2v2,other_features)
    data_with_features,start_all_other,end_all_other,inter_all_other = deal_with_data_for_curve(data_with_features, inter)



    start = -1.0#min(start_all,start_db)
    end = 1.0#max(end_all,end_db)

    import matplotlib.pyplot as plt
    #画曲线图
    fig = plt.figure()
    y = [0] * len(data_baseline)
    i = 0
    while i < len(data_baseline):
        y[i] = start + inter * i
        i += 1

    plt.plot(y,data_baseline, 'r', linewidth=2,label='data_all')
    plt.plot(y,data_with_features, 'b', linewidth=2,label='data_all_other')
    # data_all = data_all[9:] + [0,0,0,0,0,0,0,0,0,]
    # plt.plot(y,data_all, 'b', linewidth=2,label='data_all_other')
    plt.xlabel(r'similarity', fontsize=16)
    plt.ylabel(r'number', fontsize=16)
    plt.title(r'', fontsize=16)
    plt.legend(loc="upper left")
    plt.grid(True)
    # plt.savefig(('vocab_neg_compare_with_other_features_%s' % (inter)).replace('.',''))
    plt.show()

def deal_with_feature_linear(x,thr = 0.3):
    if x < thr:
        return 1/thr * x - 1
    return (0.5-thr)/(1-thr) + x / (1-thr)/2
    return x

def deal_with_baidu_linear(x,thr = 0.3):
    # if x < thr:
    #     return 1/thr * x - 1
    # return (0.5-thr)/(1-thr) + x / (1-thr)/2
    # return x
    x = 2*x - 1
    return x#sigmoid(x)
def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))

def cal_PearsonCorrelation(x,y):
    # print data
    from math import sqrt

    def multipl(a,b):
        sumofab=0.0
        for i in range(len(a)):
            temp=a[i]*b[i]
            sumofab+=temp
        return sumofab

    def corrcoef(x,y):
        n=len(x)
        #求和
        sum1=sum(x)
        sum2=sum(y)
        #求乘积之和
        sumofxy=multipl(x,y)
        #求平方和
        sumofx2 = sum([pow(i,2) for i in x])
        sumofy2 = sum([pow(j,2) for j in y])
        num=sumofxy-(float(sum1)*float(sum2)/n)
        #计算皮尔逊相关系数
        den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
        return num/den

    # x = [0,1,0,1]
    # y = [0,1,0,0.5]

    return corrcoef(x,y) #0.471404520791


def cal_Spearman(x,y):
    import scipy
    # x = [10,20,30]
    # y = [1,2,3]
    res = scipy.stats.spearmanr(x,y)
    # print res[0],res[1]
    return res[0],res[1]


def cal_cross_entropy(x,y):
    import numpy as np

    fea=np.asarray(x,np.float32)
    label=np.array(y)

    #方法一：根据公式求解
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)
    loss1=-np.sum(label*np.log(softmax(fea)))

    return loss1

def cal_correlations():

    baseline_med_pos = get_similarity_score_from_file(path = 'data/baseline score/med_7666_similarity_score_baseline_positive.txt')
    # baseline_med_neg = get_similarity_score_from_file(path = 'data\\baseline score\\med_50000_similarity_score_baseline_negative.txt')
    baseline_all_model = get_similarity_score_from_file(path = 'data/baseline score/not_only_med_50000_similarity_score_baseline_negative.txt')
    #
    data_med_pos = cal_and_get_data_res_for_medical()
    data_all_model_neg = cal_and_get_data_res_for_medical_negative(if_all_model= 1)

    import random
    thr = 7666 * 4
    baseline_all_model = random.sample(baseline_all_model, len(baseline_all_model))
    data_all_model_neg = random.sample(data_all_model_neg, len(data_all_model_neg))

    print data_med_pos[:10]
    print(len(data_med_pos))
    # # print data
    # # print data_neg
    # print len(data),len(data_neg)

    dic_baseline = {}
    dic_data = {}

    for i in baseline_all_model[:thr]:
        dic_baseline[i] = -0
    for i in baseline_med_pos:
        dic_baseline[i] = 1


    for i in data_all_model_neg[:thr]:
        dic_data[i] = -0
    for i in data_med_pos:
        dic_data[i] = 1


    baseline = sorted(dic_baseline.iteritems(),key = lambda d:d[0], reverse = 0)
    data = sorted(dic_data.iteritems(),key = lambda d:d[0], reverse = 0)

    print baseline[:100]
    print data[:100]

    write_file('res_baseline.txt',[str(i[0])+' '+str(i[1]) for i in baseline])
    write_file('res_data.txt',[str(i[0])+' '+str(i[1]) for i in data])

    labels = [-1] * thr + [1] * len(baseline_med_pos)

    print len(data_med_pos),len(data_all_model_neg)

    print 'pearson:'
    print cal_PearsonCorrelation(baseline_all_model[:thr] + baseline_med_pos, labels)
    print cal_PearsonCorrelation(data_all_model_neg[:thr] + data_med_pos, labels)
    print cal_PearsonCorrelation([i[0] for i in baseline], [i[1] for i in baseline])
    print cal_PearsonCorrelation([i[0] for i in data], [i[1] for i in data])

    print 'Spearmanr:'
    print cal_Spearman(baseline_all_model[:thr] + baseline_med_pos, labels)
    print cal_Spearman(data_all_model_neg[:thr] + data_med_pos, labels)
    print cal_Spearman([i[0] for i in baseline], [i[1] for i in baseline])
    print cal_Spearman([i[0] for i in data], [i[1] for i in data])

    print 'cross entropy:'
    print cal_cross_entropy([i[0] for i in baseline], [i[1] for i in baseline])
    print cal_cross_entropy([i[0] for i in data], [i[1] for i in data])


    print cal_cross_entropy([0,0,1,1],[0,0,1,1])
    print cal_cross_entropy([0,0,1,1],[1,1,10,10])


def ROC():


    baseline_med_pos = get_similarity_score_from_file(path = 'data/baseline score/med_7666_similarity_score_baseline_positive.txt')
    baseline_med_neg = get_similarity_score_from_file(path = 'data/baseline score/med_50000_similarity_score_baseline_negative.txt')
    # baseline_all_model_neg = get_similarity_score_from_file(path = 'data/baseline score/not_only_med_50000_similarity_score_baseline_negative.txt')
    glove_med_pos = get_similarity_score_from_file(path = 'data/baseline score/GloVe_pos.txt')
    glove_med_neg = get_similarity_score_from_file(path = 'data/baseline score/GloVe_med_neg.txt')
    # glove_all_model_neg = get_similarity_score_from_file(path='data/baseline score/GloVe_all_model_neg.txt')
    data_med_pos = cal_and_get_data_res_for_medical()
    data_med_neg = cal_and_get_data_res_for_medical_negative(if_all_model= 0)
    # data_all_model_neg = cal_and_get_data_res_for_medical_negative(if_all_model= 1)
    data_med_pos_rulebased = cal_and_get_data_res_for_medical(feature_num=3)
    data_all_model_neg_rulebased = cal_and_get_data_res_for_medical_negative(if_all_model= 1, feature_num=3)

    import numpy as np
    import sklearn.metrics as m
    import pylab as pl
    import random

    content = []

    THREASHOLD = 0.42
    while THREASHOLD <= 1:
        THREASHOLD += 0.01
        content += str(THREASHOLD),

        # for baseline
        thr = 7666 * 2
        y_true_base = [1] * 7666 + [0] * thr
        y_pred_base = baseline_med_pos + random.sample(baseline_med_neg, thr)
        # y_pred_base = baseline_med_pos + baseline_all_model_neg[-thr:]
        # print y_true_base
        # print y_pred_base
        y_cls_base = np.array([ 1 if i >= THREASHOLD else 0 for i in y_pred_base ], dtype=np.float32)
        # print list(y_cls_base).count(0)
        # print list(y_cls_base).count(1)

        # print m.classification_report(y_true_base, y_cls_base, digits= 10)
        content += m.classification_report(y_true_base, y_cls_base, digits= 10),
        # content += str(m.f1_score(y_true_base, y_cls_base)),
        print str(m.auc(y_true_base, y_cls_base)),


        # for comparison
        y_true = [1] * 7666 + [0] * thr
        y_pred = data_med_pos + random.sample(data_med_neg, thr)
        # y_pred = data_med_pos + data_all_model_neg[-thr:]
        # print y_true
        # print y_pred
        y_cls = np.array([ 1 if i >= THREASHOLD else 0 for i in y_pred ], dtype=np.float32)
        # print list(y_cls).count(0)
        # print list(y_cls).count(1)
        print m.classification_report(y_true, y_cls, digits=10)
        content += m.classification_report(y_true, y_cls, digits=10),
        # content += str(m.f1_score(y_true, y_cls)),
        print str(m.auc(y_true, y_cls)),

        # for glove
        thr_glove = 7258
        y_true_glove = [1] * thr_glove + [0] * thr_glove * 2
        y_pred_glove = glove_med_pos + random.sample(glove_med_neg, thr_glove * 2)
        # y_pred_glove = glove_med_pos + glove_all_model_neg[-thr:]
        y_cls_glove = np.array([ 1 if i >= THREASHOLD else 0 for i in y_pred_glove ], dtype=np.float32)
        print str(m.auc(y_true_glove, y_cls_glove)),


        # for rulebased
        # 给rulebased，只要前三个特征的
        y_true_rulebased = [1] * 7666 + [0] * thr
        y_pred_rulebased = data_med_pos_rulebased + random.sample(data_all_model_neg_rulebased, thr)
        # y_pred_rulebased = data_med_pos_rulebased + data_all_model_neg_rulebased[:-5000][-thr:]
        y_cls_rulebased = np.array([ 1 if i >= THREASHOLD else 0 for i in y_pred_rulebased ], dtype=np.float32)

        content += m.classification_report(y_true_rulebased, y_cls_rulebased, digits= 10),

        fpr_base, tpr_base, th_base = m.roc_curve(y_true_base, y_pred_base)
        fpr, tpr, th = m.roc_curve(y_true, y_pred)
        # fpr_glove , tpr_glove, th = m.roc_curve(y_true_glove , y_pred_glove)
        fpr_rulebased, tpr_rulebased, th_rulebased = m.roc_curve(y_true_rulebased, y_pred_rulebased)
        # fpr_ppmi, tpr_ppmi, th_ppmi = m.roc_curve(y_true_ppmi, y_pred_ppmi)

        fpr_base, tpr_base, th_base = m.roc_curve(y_true_base, y_pred_base)
        fpr, tpr, th = m.roc_curve(y_true, y_pred)
        fpr_glove , tpr_glove, th_glove = m.roc_curve(y_true_glove , y_pred_glove)
        fpr_rulebased, tpr_rulebased, th_rulebased = m.roc_curve(y_true_rulebased, y_pred_rulebased)
        ax = pl.subplot(1, 1, 1)
        ax.plot(fpr_base, tpr_base, color = 'blue')
        ax.plot(fpr, tpr, color = 'red')
        ax.plot(fpr_glove,tpr_glove, color = 'yellow')
        ax.plot(fpr_rulebased, tpr_rulebased, color = 'darkgreen')
        # ax.set_xlim([0.0, 0.25])
        # ax.set_ylim([0.5, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        ax.set_title('ROC curve, Thresh = %s.png' % (THREASHOLD))
        pl.xlabel('FP rate')
        pl.ylabel('TP rate')
        pl.savefig('imgs/ROC curve, Thresh = %s.png' % (THREASHOLD))
        # pl.show()
        pl.clf()

        precision_base, recall_base, th_base = m.precision_recall_curve(y_true_base, y_pred_base)
        precision, recall, th = m.precision_recall_curve(y_true, y_pred)
        precision_glove, recall_glove, th_glove = m.precision_recall_curve(y_true_glove, y_pred_glove)
        precision_rulebased, recall_rulebased, th_rulebased = m.precision_recall_curve(y_true_rulebased, y_pred_rulebased)
        ax = pl.subplot(1, 1, 1)
        ax.plot(recall_base, precision_base, color = 'blue')
        ax.plot(recall, precision, color = 'red')
        ax.plot(recall_glove, precision_glove, color = 'yellow')
        ax.plot(recall_rulebased, precision_rulebased, color = 'darkgreen')
        # ax.set_xlim([0.5, 1.0])
        # ax.set_ylim([0.75, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(True)
        pl.xlabel('Precision')
        pl.ylabel('Recall')
        ax.set_title('Precision recall curve, Thresh = %s.png' % (THREASHOLD))
        pl.savefig('imgs/Precision recall curve,  Thresh = %s.png' % (THREASHOLD))

        # pl.show()
        pl.clf()
        content += '\n\n'
        break

    # print content
    # write_file('res_ROC_PR.txt', content)

    # import numpy as np
    # from scipy import interp
    # import matplotlib.pyplot as plt
    #
    # from sklearn import svm, datasets
    # from sklearn.metrics import roc_curve, auc
    # from sklearn.cross_validation import StratifiedKFold
    #
    # ###############################################################################
    # # Data IO and generation,导入iris数据，做数据准备
    #
    # # import some data to play with
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # X, y = X[y != 2], y[y != 2]#去掉了label为2，label只能二分，才可以。
    # n_samples, n_features = X.shape
    # print X
    #
    # X = baseline_med_pos + baseline_all_model
    # thr = 7666 * 2
    # y = [-1] * len(baseline_med_neg[:thr]) + [1] * len(baseline_med_pos)
    # X = np.array(X)
    # y = np.array(y)
    # n_samples = len(X)
    #
    # # Add noisy features
    # random_state = np.random.RandomState(0)
    # # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    #
    # ###############################################################################
    # # Classification and ROC analysis
    # #分类，做ROC分析
    #
    # # Run classifier with cross-validation and plot ROC curves
    # #使用6折交叉验证，并且画ROC曲线
    # cv = StratifiedKFold(y, n_folds=6)
    # classifier = svm.SVC(kernel='linear', probability=True,
    #                      random_state=random_state)#注意这里，probability=True,需要，不然预测的时候会出现异常。另外rbf核效果更好些。
    #
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    #
    # for i, (train, test) in enumerate(cv):
    #     #通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    #     probas_ = X#classifier.fit(X[train], y[train]).predict_proba(X[test])
    # #    print set(y[train])                     #set([0,1]) 即label有两个类别
    # #    print len(X[train]),len(X[test])        #训练集有84个，测试集有16个
    # #    print "++",probas_                      #predict_proba()函数输出的是测试集在lael各类别上的置信度，
    # #    #在哪个类别上的置信度高，则分为哪类
    #     # Compute ROC curve and area the curve
    #     #通过roc_curve()函数，求出fpr和tpr，以及阈值
    #     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    #     mean_tpr += interp(mean_fpr, fpr, tpr)			#对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    #     mean_tpr[0] = 0.0 								#初始处为0
    #     roc_auc = auc(fpr, tpr)
    #     #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    #     plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    #
    # #画对角线
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #
    # mean_tpr /= len(cv) 					#在mean_fpr100个点，每个点处插值插值多次取平均
    # mean_tpr[-1] = 1.0 						#坐标最后一个点为（1,1）
    # mean_auc = auc(mean_fpr, mean_tpr)		#计算平均AUC值
    # #画平均ROC曲线
    # #print mean_fpr,len(mean_fpr)
    # #print mean_tpr
    # plt.plot(mean_fpr, mean_tpr, 'k--',
    #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

def get_model_from_txt_and_cal_similarity(path = ''):
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load_word2vec_format('model/Gvectors.txt', binary=False)
    # print(word_vectors.vocab.keys[0])
    # print(word_vectors[u'感冒'])
    # print(word_vectors.similarity(u'感冒',u'发烧'))
    filename_pos = 'data/syns/medical_syns_7666.txt'
    f = open(filename_pos, 'r')
    count = 0
    total = 0
    res = []
    score = []
    for line in f:
        total += 1
        line = line.strip()
        # print(line)
        line = line.split('|')
        e1 = line[0]
        e2 = line[1]
        res += (e1, e2),
        try:
            score += word_vectors.similarity(e1.decode(),e2.decode()),
        except:
            # print(e1+' '+e2)
            # count += 1
            continue
        # print score
    score = [str(i) for i in score]
    write_file('data/baseline score/GloVe_pos.txt', score)

    filename_neg = 'data/neg_syns/word_from_model_negative_100w.txt'
    f = open(filename_neg, 'r')
    total_neg = 0
    res_neg = []
    score_neg = []
    for line in f:
        total_neg += 1
        line = line.strip()
        line = line.split('|')
        e1_neg = line[0]
        e2_neg = line[1]
        res_neg += (e1_neg, e2_neg),
        try:
            score_neg += word_vectors.similarity(e1_neg.decode(),e2_neg.decode()),
        except:
            continue
    score_neg.sort()
    score_neg = [str(i) for i in score_neg]
    write_file('data/baseline score/GloVe_neg.txt', score_neg)

    filename_med_neg = 'data/neg_syns/med_negative_100w.txt'
    f = open(filename_med_neg, 'r')
    total_med_neg = 0
    res_med_neg = []
    score_med_neg = []
    for line in f:
        total_neg += 1
        line = line.strip()
        line = line.split('|')
        e1_med_neg = line[0]
        e2_med_neg = line[1]
        res_med_neg += (e1_med_neg, e2_med_neg),
        try:
            score_med_neg += word_vectors.similarity(e1_med_neg.decode(), e2_med_neg.decode()),
        except:
            continue
    score_med_neg.sort()
    score_med_neg = [str(i) for i in score_med_neg]
    write_file('data/baseline score/GloVe_med_neg.txt', score_med_neg)

# get_model_from_txt_and_cal_similarity()



# sentences = loadFileToTrain('data/seged text/text_EMR_seged.txt')
# time.sleep(10000)
# Load_file_and_train('data/seged text/text_EMR_seged.txt','model_EMR')
# Load_file_and_train('data/seged text/DB_ALL_seged.txt','model_DB')
# Load_file_and_train('data/seged text/RE_ALL_seged.txt','model_RE')
# Load_file_and_train('data/seged text/baike_ALL_seged.txt','model_Baike')
# Load_file_and_train('data/seged text/neike_ALL_seged.txt','model_Neike')

# incremental_train('data/seged text/text_EMR_seged.txt','model_EMR')
# incremental_train('data/seged text/wiki_ALL_seged.txt','model_EMR_iter_1')

# analyze_txt()


# get_words_without_dups()


# load_files_and_train('model_DB_neike_abs_RE_wiki')


# get_neg_data_from_model()



#[][][][][][
# cal_and_draw_curve(0.03)
# cal_and_draw_curve_and_get_percent(0.03)



# baidu_dic = features.getSearchScoreFromCSV('data/baidu_7666.csv')


# cal_and_draw_curve_with_other_features_for_medical(0.03)
# cal_and_draw_curve_with_other_features_for_medical_negative(if_all_model= 0)


ROC()
# cal_correlations()
#get_model_from_txt_and_cal_similarity()

        # write_word2vec_similarity_to_file()
# write_word2vec_to_file()


# baseline_med = get_similarity_score_from_file(path = 'D:\python\WordSimilarity\\data\\baseline score\\med_7666_similarity_score_baseline_positive.txt')
# baseline_med = get_similarity_score_from_file(path = 'D:\python\WordSimilarity\\data\\baseline score\\med_50000_similarity_score_baseline_negative.txt')
# print baseline_med
# print len(baseline_med)


# _,_,w2v_med_1,w2v_med_2 = get_word2vec_from_file("D:\python\WordSimilarity\data\data_vector\\word2vec_med_7666_positive.txt")
# for i in w2v_med_1[:10]:
#     print i
# for i in w2v_med_2[:10]:
#     print i

# res = []
# res_all = get_words_not_OOV(filename='data/neg_syns/med_negative_100w.txt')
# for i in res_all:
#     res += i[0] + '|' + i[1],
# write_file('data/neg_syns/med_negative_100w_De_OOV.txt',res)



# x = 0
# while x <= 1:
#     print(x,deal_with_feature_linear(x,0.35))
#     x += 0.01





# not_oov = get_words_not_OOV()
# print(len(not_oov))
# res = []
# for i,j in not_oov:
#     res += i+'|'+j,
# write_file('data/syns/medical_syns_7666.txt',res)



# model_original = gensim.models.Word2Vec.load('model/model_EMR_iter_1')
# model_test = gensim.models.Word2Vec.load('model/test')
# model_DB = gensim.models.Word2Vec.load('model/model_DB_iter_1')
#
#
# model = gensim.models.Word2Vec.load('model/model_DB_neike_abs_RE_wiki')
# # print model_test.similarity('肺大疱','肺大泡')
# print model.similarity('感冒','发烧')
#
# for i in model.most_similar('冠心病'):
#     print i[0],i[1]
#
# print len(model.vocab)
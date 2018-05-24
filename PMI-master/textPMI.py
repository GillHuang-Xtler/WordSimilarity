#coding=utf-8
__author__ = 'root'
from PMI import *
import os
from extract import extract


def read_file_and_get_text_increamentally(path):
    documents = []
    f = open(path, 'r',encoding='utf8')
    count = 0
    test = 0
    for data in f:
        if not data:
            break
        if count % 50000 == 0:
            print(count)
        # if count > 10000:
        #     break
        data = data.strip()
        data = data.split(' ')
        if len(data) <= 2:
            continue
        documents.append((set(data)))
        count += 1
    print('READ Done')
    return documents



def write_file(path,content):
    with open(path, mode='a') as f:
        for i in content:
            try:
                f.writelines(i + '\n')
            except:
                print(i,'??????')
                continue
        f.close()


def get_positive_samples():
    f = open('..\\data\\syns\\medical_syns_7666.txt','r',encoding='utf8')
    res = []
    for line in f:
        line = line.strip()
        line = line.split('|')
        # print(line)
        res += (line[0],line[1]),
    return res

def get_negative_samples():
    f = open('..\\data\\neg_syns\\word_from_model_negative_100w.txt','r',encoding='utf8')
    res = []
    for line in f:
        line = line.strip()
        line = line.split('|')
        # print(line)
        res += (line[0],line[1]),
    return res



def read_file_and_cal_PMI_and_write_to_file():
    documents = read_file_and_get_text_increamentally('..\\data\\seged text\\wiki_ALL_seged.txt')
    print(len(documents))
    documents += read_file_and_get_text_increamentally('..\\data\\seged text\\baike_ALL_seged.txt')
    print(len(documents))
    documents += read_file_and_get_text_increamentally('..\\data\\seged text\\DB_ALL_seged.txt')
    print(len(documents))
    documents += read_file_and_get_text_increamentally('..\\data\\seged text\\neike_ALL_seged.txt')
    print(len(documents))
    pm = PMI(documents)

    pos = get_positive_samples()
    neg = get_negative_samples()[:50000]

    res_pos = []
    for e1,e2 in pos:
        tmp = ''
        tmp += e1 + '|' + e2 + '|' + str(pm.get_pmi(e1,e2))
        print(tmp)
        res_pos += tmp,
        write_file('res_pos.txt',res_pos)
        res_pos = []

    res_neg = []
    for e1,e2 in neg:
        tmp = ''
        tmp += e1 + '|' + e2 + '|' + str(pm.get_pmi(e1,e2))
        print(tmp)
        res_neg += tmp,
        write_file('res_neg.txt',res_neg)
        res_neg = []

    # pmi = pm.get_pmi('病毒','传染病')
    # print(pmi)
    # pmi = pm.get_pmi('骨膜','食管中段憩室')
    # print(pmi)
    # pmi = pm.get_pmi('感冒','发烧')
    # print(pmi)

def read_res_and_get_similarity_score():

    f = open('res_pos.txt','r')
    res = []
    for line in f:
        line = line.strip()
        line = line.split('|')
        print(line)
        if line[2] == 'None':
            res += '0'
        else:
            res += line[2],#(line[0],line[1]),
    res = [float(i) for i in res]
    maxx = max(res)
    res = [str(i/maxx) for i in res]

    return res
def read_res_and_get_similarity_score_neg():

    f = open('res_neg.txt','r')
    res = []
    for line in f:
        line = line.strip()
        line = line.split('|')
        print(line)
        if len(line) > 3:
            continue
        if line[2] == 'None':
            continue
            res += '0'
        else:
            res += line[2],#(line[0],line[1]),
    print(res)
    res = [float(i) for i in res]
    # maxx = max(res)
    res = [str(i) for i in res]

    return res

if __name__ == '__main__':
    # score = read_res_and_get_similarity_score()
    # print(score)
    # write_file('med_7666_similarity_socre_of_PPMI.txt', score)

    import random
    for i in range(100):
        print(random.random() - 0.5)

    # score = read_res_and_get_similarity_score_neg()
    # print(score)
    # write_file('med_neg_similarity_socre_of_PPMI_origin.txt', score)

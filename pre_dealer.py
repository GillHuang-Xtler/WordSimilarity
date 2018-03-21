#-*- coding: UTF-8 -*-

import jieba
import os

import codecs
import random


class Seg(object):
    stopwords = []
    stopword_filepath="./data//stopword.txt"

    def __init__(self):
        self.read_in_stopword()
        # jieba.load_userdict('data/Medical_items.txt')
        jieba.load_userdict('data/Medical_items_ALL_ALL_from_KG.txt')

        # 读入用户字典
        for line in open("data/Medical_items_ALL_ALL_from_KG.txt", mode='r', encoding='utf-8'):
            line = line.rstrip()
            line = line.split(' ')
            # 保留词长度 <= 12的
            if len(line[0]) > 12:
                continue
            jieba.add_word(line[0])
        print('Manually ADDED')
        # self.

    def read_in_stopword(self):
        file_obj = codecs.open(self.stopword_filepath,'r','utf-8')
        while True:
            line = file_obj.readline()
            #lineList = line.split(u'。')

            line=line.strip('\r\n')
            if not line:
                break
            self.stopwords.append(line)
        file_obj.close()

        self.stopwords = set(self.stopwords)

    def cut(self,sentence,stopword=True):
        seg_list = jieba.cut(sentence,cut_all=False,HMM=False)

        results = []
        for seg in seg_list:
            if seg in self.stopwords and stopword:
                continue
            results.append(seg)

        return results



def write_file(path,content):
    with open(path, mode='a', encoding='utf-8') as f:
        for i in content:
            f.writelines(i + '\n')
        f.close()



def get_all_sentence_without_duplicates():

    res = set()

    count = 0
    rootdir = 'D:\python\Baidu_KG\data\output-多分类-2017-6-8\output'
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        # if os.path.isfile(path):
        print(path)

        for line in open(path, mode='r', encoding='utf-8'):
            count += 1
            # print('ori:',line.split()[3])
            try:
                tmp = line.split()[3].split('@@@@@')
            except:
                continue
            if len(tmp) < 2:
                # print(tmp[0])
                res.add(tmp[0])
                continue
            # print(tmp[0])
            # print(tmp[1])
            res.add(tmp[0])
            res.add(tmp[1])
            if count % 10000 == 0:
                print(count)
                print(len(res)/2)
            # if count >= 10:
            #     return

    write_file('data\medical_text_ALL.txt',res)


def seg_sentences_for_RE():

    seg = Seg()
    path = 'data\\original text\\medical_text_ALL_from_RE.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            print(count,len(res))
            write_file('data\seged text\RE_ALL_seged.txt',res)
            res = []
    print(count)
    print(count,len(res))
    write_file('data\seged text\RE_ALL_seged.txt',res)

def seg_sentences_for_Wiki():

    seg = Seg()
    path = 'data\数据 wiki 病例 原始\wiki_cn'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 2:
            continue
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            print(count,len(res))
            write_file('data\seged text\wiki_ALL_seged.txt',res)
            res = []
    print(count)
    write_file('data\seged text\wiki_ALL_seged.txt',res)

def seg_sentences_for_EMR():

    seg = Seg()
    path = 'data\数据 wiki 病例 原始\Quanke_med.dat'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 4:
            continue
        if '2016-' in line:
            continue
        if '2012-' in line:
            continue
        if '√' in line or '□' in line:
            continue
        if 'BORDER-RIGHT' in line or '.table1' in line or '.td1 {' in line:
            continue
        if '中医药大学第一附属医院' in line:
            continue
        line = line.replace('\xa0','')
        line = line.replace('\u3000\u3000','')
        line = line.replace('\u3000',' ')
        line = line.strip().lstrip().rstrip()
        if len(line) <= 12:
            continue
        count += 1
        # print(line)
        # print(len(line))
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            # return
            print(count,len(res))
            write_file('data\\seged text\\text_EMR_seged.txt',res)
            res = []
            # return
    print(count)


def seg_sentences_for_DB():

    seg = Seg()
    path = 'data\\original text\\text_from_DB_Baidu.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 2:
            continue
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(line)
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 10000 == 0:
            # return
            print(count,len(res))
            write_file('data\seged text\DB_ALL_seged.txt',res)
            res = []
    print(count)
    write_file('data\seged text\DB_ALL_seged.txt',res)

def seg_sentences_for_baike_abstract():

    seg = Seg()
    path = 'data\\original text\\abstract_baidu.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 2:
            continue
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(line)
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            # return
            print(count,len(res))
            write_file('data\seged text\\baike_ALL_seged.txt',res)
            res = []
    path = 'data\\original text\\abstract_hudong.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 2:
            continue
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(line)
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            # return
            print(count,len(res))
            write_file('data\seged text\\baike_ALL_seged.txt',res)
            res = []
    print(count)
    write_file('data\seged text\\baike_ALL_seged.txt',res)


def seg_sentences_for_neike():

    seg = Seg()
    path = 'data\\original text\\neike.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        if len(line) < 2:
            continue
        count += 1
        line = line.strip()
        line = line.lstrip()
        line = line.rstrip()
        # print(line)
        # print(seg.cut(line))
        res += ' '.join(seg.cut(line)),
        # if count >= 10:
        #     break
        if count % 1000 == 0:
            # return
            print(count,len(res))
            write_file('data\seged text\\neike_ALL_seged.txt',res)
            res = []
    print(count)
    write_file('data\seged text\\neike_ALL_seged.txt',res)




def DB_connector():

    import pymysql
    conn_local = pymysql.connect(host='localhost', user='root', passwd='root', db='baidu_kg_0322', port=3306,charset='utf8')
    cur_local = conn_local.cursor()

    print(conn_local)
    return conn_local,cur_local

def DB_get_description_from_DB(conn,cur):

    res = []


    table = ['food', 'laboratory_test'] #'medicine','recipes'
    sql = 'select %s_name,%s_description from %s' % (table[0],table[0],table[0])

    cur.execute(sql)
    for i in cur.fetchall():
        if i[1] is None:
            continue
        tmp = i[0] + ' ' + i[1].lstrip().rstrip()
        res += tmp,
    sql = 'select %s_name,%s_description from %s' % (table[1],table[1],table[1])

    cur.execute(sql)
    for i in cur.fetchall():
        if i[1] is None:
            continue
        tmp = i[0] + ' ' + i[1].lstrip().rstrip()
        res += tmp,



    table = 'medicine'
    sql = 'select %s_name, function, announcement, pharmacological_action from %s' % (table, table)

    cur.execute(sql)
    for i in cur.fetchall():
        i = list(i)
        if i[1] is None:
            i[1] = ''
        if i[2] is None:
            i[2] = ''
        if i[3] is None:
            i[3] = ''

        tmp = i[0] + ' '
        tmp += i[1].lstrip().rstrip() + ' '
        tmp += i[2].lstrip().rstrip() + ' '
        tmp += i[3].lstrip().rstrip() + ' '
        if len(tmp.split()) == 1:
            continue
        # print(tmp)
        res += tmp,

    table = 'recipes'
    sql = 'select %s_name, cooking_analysis from %s' % (table, table)

    cur.execute(sql)
    for i in cur.fetchall():
        if i[1] is '':
            continue
        tmp = i[0] + ' ' + i[1].lstrip().rstrip()
        # print(tmp)
        res += tmp,



    table = 'disease'
    sql = 'select %s_name, description, cause, precaution, clinical_manifestation, ware_issues, could_eat_food_txt, could_not_eat_food_txt, could_eat_dish_txt, food_txt, disease_txt, lab_test_txt, drug_txt, symptom_txt   from %s' % (table, table)

    cur.execute(sql)
    for i in cur.fetchall():
        i = list(i)
        tmp = ''
        for j in i:
            if j is None:
                continue
            tmp += j.lstrip().rstrip() + ' '

        if len(tmp.split()) == 1:
            continue
        # print(tmp)
        res += tmp,
    # write_file('data\\text_from_DB_Baidu.txt', res)


    table = 'symptom'
    sql = 'select %s_name, description, pathogeny, prevention, examine, antidiastole, could_not_eat_food_txt, could_not_eat_food_items, could_eat_food_txt, could_eat_food_items, raw_data  from %s' % (table, table)

    cur.execute(sql)
    for i in cur.fetchall():
        i = list(i)
        tmp = ''
        for j in i:
            if j is None:
                continue
            tmp += j.lstrip().rstrip() + ' '

        if len(tmp.split()) == 1:
            continue
        # print(tmp)
        res += tmp,
    # write_file('data\\text_from_DB_Baidu.txt', res)

    table = 'surgery'
    sql = 'select %s_name, description, surgery_notice, surgery_after_treat, fit_symptoms, sequela, surgery_notice_items, preparation, contraindication, surgery_steps, surgery_influence, surgery_diet, complication, raw_data   from %s' % (table, table)

    cur.execute(sql)
    for i in cur.fetchall():
        i = list(i)
        tmp = ''
        for j in i:
            if j is None:
                continue
            tmp += j.lstrip().rstrip() + ' '

        if len(tmp.split()) == 1:
            continue
        # print(tmp)
        res += tmp,



    write_file('data\\text_from_DB_Baidu.txt', res)




def DB_get_alias_from_DB(conn,cur):

    dic = {}

    i = 0

    table = ['food', 'disease','surgery','symptom'] #'medicine','recipes'

    for i in range(4):
        sql = 'select main_name, alias_name from alias_%s' % (table[i])

        cur.execute(sql)
        for i in cur.fetchall():
            # print(i[0],i[1])
            dic[i[0]] = dic.get(i[0],[]) + [i[1]]
    for k,v in dic.items():
        print(k,v)
    print(len(dic))

    res = []

    for k,v in dic.items():
        # print(k,v)
        res += k + ' ' + ' '.join(v),

    print(res)
    write_file('data\\alias_from_Baidu_DB.txt',res)



def get_syns_from_text():
    path = 'data\\syns\\alias_from_Baidu_DB.txt'
    count = 0
    res = []
    for line in open(path, mode='r', encoding='utf-8'):
        # print(line)
        line = line.split()
        # print(line)
        # for i in res:
        #     for j in line:
        #         if j in i:
        #             for jj in line:
        #                 i.add(jj)
        #             continue
        tmp = set(line)
        if len(tmp) == 1:
            continue
        print(tmp)
        res += tmp,


    path = 'data\\syns\\syn_all.txt'
    for line in open(path, mode='r', encoding='utf-8'):
        # print(line)
        line = line.rstrip()
        line = line.split(' ')
        # print(line)
        # for i in res:
        #     for j in line:
        #         if j in i:
        #             for jj in line:
        #                 i.add(jj)
        #             continue
        tmp = set(line)
        if len(tmp) == 1:
            continue
        print(tmp)
        res += tmp,
    # for i in res:
    #     print(i)
    print(len(res))
    return res


def get_syns_from_data_YKQ():
    dic = {}

    path = 'data\\syns\\out2.txt'
    for line in open(path, mode='r', encoding='utf-8'):
        # print(line)
        line = line.rstrip()
        line = line.split(',')
        print(line)
        dic[line[1]] = dic.get(line[1],[]) + line[2].split('|')
    path = 'data\\syns\\out3.txt'
    for line in open(path, mode='r', encoding='utf-8'):
        # print(line)
        line = line.rstrip()
        line = line.split(',')
        print(line)
        dic[line[1]] = dic.get(line[1],[]) + line[2].split('|')

    # for i in res:
    #     print(i)
    for k,v in dic.items():
        # print(k,v)
        v = list(set(v))
        print(k,v)
    # print(len(dic))
    return dic

def write_alias_from_set_to_file(s):
    res = []
    for i in s:
        i = list(i)
        if len(i) == 1:
            continue
        # if len(i) > 10:
        #     print(i)
        for ii in range(len(i)):
            for jj in range(ii+1,len(i)):
                # pass
                # print(ii,jj)
                print( i[ii] + '|' + i[jj])
                res += i[ii] + '|' + i[jj],
    # print(res[0:12])
    write_file('data\\syns\\syns_01.txt',res)


def write_alias_from_dic_to_file(dic):
    res = []
    for k,v in dic.items():
        for j in v:
            res += k + '|' + j,
    write_file('data\\syns\\syns_02.txt',res)


def get_words_without_dups():
    res = set()
    f = open('data/syns/syns_ALL_1214.txt','r')
    count = 0
    for line in f:
        print(line)
        res.add(line)

    res = list(res)
    write_file('data/syns/syns_ALL_1214_without_dups.txt',res)


def get_items_from_file_as_set(path = 'data\\syns\\syns_ALL_1214_without_dups.txt'):


    count = 0
    res = set()
    for line in open(path, mode='r', encoding='utf-8'):
        line = line.strip()
        # print(line)
        line = line.split('|')
        res.add(line[0])
        res.add(line[1])

    # print(res)
    # print(len(res))
    return res
def get_items_from_file_as_list(path = 'data\\syns\\syns_ALL_1214_without_dups.txt'):


    count = 0
    res = set()
    for line in open(path, mode='r', encoding='utf-8'):
        line = line.strip()
        # print(line)
        line = line.split('|')
        # res += line,
        res.add((line[0],line[1]))

    # print(res)
    # print(len(res))
    return res

def update_medical_dic(s):
    print(s)
    s = list(s)
    res = []
    for i in s:
        i = i + ' n 1'
        res += i,
    write_file('data/Medical_items_ALL_with_syns_1215.txt',res)





def get_general_syns():

    path = 'data\\syns\\general_synonym.txt'
    count = 0
    res = []
    all_words = []
    for line in open(path, mode='r', encoding='utf-8'):
        line = line.strip()
        # print line
        line = line.split()
        # print line
        res += line,
        all_words += line
    return res,all_words










def get_neg_syns_from_txt():
    # gen_syns, all_gen_words = get_general_syns()
    # print(gen_syns[0][0])
    # print(all_gen_words[0])
    #
    # med_words_set = get_items_from_file_as_set()
    # med_syns_set = get_items_from_file_as_list()
    #
    # # print(med_syns_list[0])
    #
    # med_words_list = list(med_words_set)
    # # print(med_words_list)
    #
    # total_med = len(med_words_list)

    word_set = get_items_from_file_as_set()
    word_list = []

    print(len(word_set))

    path = 'data\\neg_syns\\vocab_from_model.txt'
    for line in open(path, mode='r', encoding='utf-8'):
        line = line.rstrip()
        word_list += line,


    total = len(word_list)

    res = []
    for ii in range(1000000):
        i = random.random() * 987654321
        j = random.random() * 987654321
        i = int(i) % total
        j = int(j) % total
        while j == i:
            j = random.random() * 987654321
            j = int(j) % total
        # print(med_words_list[i],med_words_list[j])
        e1, e2 = word_list[i],word_list[j]
        if ii % 10000 == 0:
            print(ii)
        if (e1,e2) in word_set or (e2,e1) in word_set:
            print(e1,e2)
            continue
        res += e1 + '|' + e2,

    write_file('data/word_from_model_negative_100w.txt', res)



get_neg_syns_from_txt()


# s = get_items_from_file()
# update_medical_dic(s)


# get_all_sentence_without_duplicates()
# seg_sentences_for_EMR()
# seg_sentences_for_Wiki()
# seg_sentences_for_RE()
# seg_sentences_for_DB()
# seg_sentences_for_baike_abstract()
# seg_sentences_for_neike()

# s = get_syns_from_text()
# write_alias_from_set_to_file(s)

# dic = get_syns_from_data_YKQ()
# write_alias_from_dic_to_file(dic)


# get_words_without_dups()


# conn, cur = DB_connector()
# DB_get_description_from_DB(conn, cur)
# conn.close()


# conn, cur = DB_connector()
# DB_get_alias_from_DB(conn, cur)
# conn.close()



# path = 'data\数据 wiki 病例 原始\wiki_cn'
# path = 'data\wiki_ALL_seged.txt'
# for line in open(path, mode='r', encoding='utf-8'):
#     print(line)
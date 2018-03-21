# -*- coding: utf-8 -*-

import re
import csv
import codecs
import urllib2
from bs4 import BeautifulSoup
from pyquery import PyQuery as pq
import time

class BaiduSearchRequest(object):

    def post(self, queryStr):

        timeout = 5
        queryStr = urllib2.quote(queryStr)
        url = 'https://www.baidu.com/s?wd=%s' % queryStr
        print url

        request = urllib2.Request(url)
        #伪装HTTP请求
        request.add_header('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')
        request.add_header('connection','keep-alive')
        request.add_header('referer', url)
        # request.add_header('Accept-Encoding', 'gzip')  # gzip可提高传输速率，但占用计算资源
        try:
            response = urllib2.urlopen(request, timeout = 10)
            html = response.read()
        except:
            print 'not catch'
            return None


        # response = urllib2.urlopen(request)

        return html
        # print html
        #results = self.extractSearchResults(html)

    def anlysis_from_html(self,html_doc):
        soup = BeautifulSoup(html_doc, 'html.parser')
        li = soup.find('div',class_ = 'nums')
        try:
            print li.text
        except:
            return '0'
        res = re.findall(r"[0-9]",li.text)
        res = "".join(res)
        return res

    def getResultNum(self,word1):

        html = self.post(word1)
        while html == None:
            html = self.post(word1)

        res = self.anlysis_from_html(html)
        return int(res)

    def main(self,word1,word2):
        res1 = self.getResultNum(word1)
        res2 = self.getResultNum(word2)
        res_both = self.getResultNum(word1+' '+word2)

        return res1,res2,res_both


def read_csv(filepath):
    llist = []
    reader = csv.reader(file(filepath, 'rb'))
    for line in reader:
        llist.append(line)

    return llist

def write_file(filepath,content):
    writer = csv.writer(file(filepath, 'a'))
    writer.writerow(content)

def get_medical_positive():
    baidu = BaiduSearchRequest()
    # print baidu.main('电脑','车')

    filepath = 'data/syns/medical_syns_7666.txt'


    # f = open('data/syns/syns_ALL_1214.txt','r')
    f = open(filepath,'r')
    count = 0
    total = 0
    res = []
    for line in f:
        total += 1
        line = line.strip()
        # print line
        e1 = line.split('|')[0]
        e2 = line.split('|')[1]
        print e1,e2
        res = baidu.main(e1,e2)

        with open('data/baidu_7666.csv', mode='a') as f:
            f.writelines(e1+'|'+e2+'|%s|%s|%s' % (res[0],res[1],res[2]) + '\n')
        f.close()
        # write_file('data/baidu_7666.csv',(e1+'|'+e2+'|%s|%s|%s' % (res[0],res[1],res[2])))


def get_nagetive():
    baidu = BaiduSearchRequest()
    # print baidu.main('电脑','车')

    # filepath = 'data/neg_syns/word_from_model_negative_100w.txt'
    filepath = 'data/neg_syns/med_negative_100w_De_OOV.txt'


    # f = open('data/syns/syns_ALL_1214.txt','r')
    f = open(filepath,'r')
    count = 0
    total = 0
    res = []
    for line in f:
        print total
        if total >= 100000:
            break
        total += 1
        if total < 1900:
            continue
        line = line.strip()
        # print line
        e1 = line.split('|')[0]
        e2 = line.split('|')[1]
        print e1,e2
        if e1 == '' or e2 == '':
            continue
        res = baidu.main(e1,e2)

        # with open('data/baidu_nagetive_all_model_vocab_1221.csv', mode='a') as f:
        with open('data/baidu_nagetive_medical_1221.csv', mode='a') as f:
            f.writelines(e1+'|'+e2+'|%s|%s|%s' % (res[0],res[1],res[2]) + '\n')
        f.close()
        # write_file('data/baidu_7666.csv',(e1+'|'+e2+'|%s|%s|%s' % (res[0],res[1],res[2])))



if __name__ == '__main__':
    get_nagetive()




    # # baidu.main('小儿休克','婴幼儿休克')
    # baidu.main('触诊无肺动脉关闭感','心血管造影见双球征')

    # zh_w2v_model = read_in_model(zh_w2v_model_filepath)
    # en_w2v_model = read_in_model(en_w2v_model_filepath)

    # radios = [1]
    # # set_filepath = 'data/%sset_1to%d.csv'
    # # features_set_filepath = 'data/%sset_1to%d_svm.csv'
    #
    # res_file = 'res/baidu_score_1to1_1014.csv'
    #
    # bd = BaiduSearchRequest()
    #
    # for radio in radios:
    #     trainset_filepath = 'data/trainset_1014.csv'#set_filepath % ('train',radio)
    #     testset_filepath = 'data/testset_1014.csv'#set_filepath % ('test',radio)
    #
    #     # 读入word pair
    #     trainset = read_csv(trainset_filepath)
    #     testset = read_csv(testset_filepath)
    #
    #     for i in trainset:
    #         res1,res2,res_both = bd.main(i[0],i[1])
    #         line = [i[0],i[1], str(res1) + '|' + str(res2) + '|' + str(res_both)[:-1]]
    #         # print line
    #         write_file(res_file,line)
    #
    #     for i in testset:
    #         res1,res2,res_both = bd.main(i[0],i[1])
    #         line = [i[0],i[1], str(res1) + '|' + str(res2) + '|' + str(res_both)[:-1]]
    #         # print line
    #         write_file(res_file,line)
    #
    #     # features_trainset_filepath = features_set_filepath % ('train',radio)
    #     # features_testset_filepath = features_set_filepath % ('test',radio)
    #     #
    #     # # 保存结果
    #     # write_file(features_trainset_filepath,trainset_features)
    #     # write_file(features_testset_filepath,testset_features)
    #



# trans.getSymptoms()
# trans.getSymptomsFromCSV('.\data\\merge.csv')



# # -*- coding: utf-8 -*-
#
# import re
# import csv
# import codecs
# import urllib2
# from bs4 import BeautifulSoup
# from pyquery import PyQuery as pq
# import time
# class BaiduSearchRequest(object):
#
#     def post(self, queryStr):
#         # print url
#         try:
#             timeout = 5
#             queryStr = urllib2.quote(queryStr)
#             url = 'https://www.baidu.com/s?wd=%s' % queryStr
#             print url
#
#             request = urllib2.Request(url)
#             #伪装HTTP请求
#             request.add_header('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')
#             request.add_header('connection','keep-alive')
#             request.add_header('referer', url)
#             # request.add_header('Accept-Encoding', 'gzip')  # gzip可提高传输速率，但占用计算资源
#             response = urllib2.urlopen(request, timeout = 5)
#
#             # response = urllib2.urlopen(request)
#             html = response.read()
#             return html
#             # print html
#             #results = self.extractSearchResults(html)
#         except Exception as e:
#             print 'URL Request Error:', e
#             return None
#
#     def anlysis_from_html(self,html_doc):
#         soup = BeautifulSoup(html_doc, 'html.parser')
#         li = soup.find('div',class_ = 'nums')
#         print li.text
#         res = re.findall(r"[0-9]",li.text)
#         res = "".join(res)
#         return res
#
#     def getResultNum(self,word1):
#
#         html = self.post(word1)# + ' ' + word2)
#         res = self.anlysis_from_html(html)
#         return res
#
#     def main(self,word1,word2):
#         res1 = self.getResultNum(word1)
#         res2 = self.getResultNum(word2)
#         res_both = self.getResultNum(word1+' '+word2)
#
#         print res1,res2,res_both
#
#
#
#
#
# baidu = BaiduSearchRequest()
# # baidu.main('电脑','车')
# # baidu.main('小儿休克','婴幼儿休克')
# baidu.main('触诊无肺动脉关闭感','心血管造影见双球征')
#
# # trans.getSymptoms()
# # trans.getSymptomsFromCSV('.\data\\merge.csv')
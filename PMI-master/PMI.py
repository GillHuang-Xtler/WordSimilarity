# coding=utf-8
import math
class PMI:
    def __init__(self, document):
        self.document = document
        self.pmi = {}
        self.miniprobability = float(1.0) / document.__len__() - 1
        self.minitogether = float(0)/ document.__len__()
        # self.set_word = self.getset_word()
        # self.word_frq = self.get_dict_frq_word()
        print('freq done')

    def calcularprobability(self, document, wordlist):

        """
        :param document:
        :param wordlist:
        :function : 计算单词的document frequency
        :return: document frequency
        """

        total = document.__len__()
        number = 0
        for doc in document:
            if set(wordlist).issubset(doc):
                number += 1
        percent = float(number)/total
        return percent

    def togetherprobablity(self, document, wordlist1, wordlist2):

        """
        :param document:
        :param wordlist1:
        :param wordlist2:
        :function: 计算单词的共现概率
        :return:共现概率
        """

        joinwordlist = wordlist1 + wordlist2
        percent = self.calcularprobability(document, joinwordlist)
        return percent

    def getset_word(self):

        """
        :function: 得到document中的词语词典
        :return: 词语词典
        """

        set_word = set()
        for doc in self.document:
            for word in doc:
                set_word.add(word)
        return set_word


    def get_pmi(self,e1,e2):
        """
        function:返回符合阈值的pmi列表
        :return:pmi列表
        """

        f1 = self.calcularprobability(self.document,[e1])
        f2 = self.calcularprobability(self.document,[e2])
        together_probability = self.togetherprobablity(self.document,[e1],[e2])
        # print(f1,f2,together_probability)

        if f1 * f2 <= 0.0 or together_probability <= 0:
            return 'None'
        return math.log(together_probability/(f1 * f2))#

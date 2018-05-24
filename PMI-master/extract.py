#coding=utf-8
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re

class Extractkeys:
 
    def removeEmoji(self, sentence):
        return re.sub('\[.*?\]', '', sentence)

    def CutWithPartOfSpeech(self, sentence):
        sentence = self.removeEmoji(sentence)
        words =jieba.cut(sentence)
        wordlist=[]
        for word in words:
            wordlist.append(word)
        return wordlist

    def ExtractWord(self,wordlist):
        sentence = ','.join(wordlist)
        words = jieba.analyse.extract_tags(sentence,5)
        wordlist = []
        for w in words:
            wordlist.append(w)
        return wordlist

    def RemoveStopWord(self,wordlist):
        stopWords = self.GetStopWords()
        keywords = []
        for word in wordlist:
            if word not in stopWords:
                keywords.append(word)
        return keywords

def extract(text):
    ek = Extractkeys()
    wordlist = ek.CutWithPartOfSpeech(text)
    extractwords = ek.ExtractWord(wordlist)
    return extractwords



#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/11/4 12:53
#

import numpy as np
from keras import Sequential, Input
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.optimizers import Adadelta
from keras_contrib.layers import CRF
from keras_preprocessing.sequence import pad_sequences

from stanfordcorenlp import StanfordCoreNLP
import spacy
import neuralcoref


# print(doc._.coref_clusters)

# nlp = StanfordCoreNLP(r'E:\nlpData\stanford-corenlp-full-2018-02-27')
# props = {'annotators': 'coref, ssplit', 'pipelineLanguage': 'en'}
# sentence = 'My sister has a son and she loves him.'
# def corefs(sentence, source):
#     output = nlp.annotate(sentence, properties={'annotators':'coref','outputFormat':'json','ner.useSUTime':'false'})
#     out = json.loads(output)
#     res = []
#     text  = []
#     for key, value in out['corefs'].items():
#         print(value)
#         same = []
#         for i in value:
#             # 因为source长度都为1
#             # if i['endIndex']- i['startIndex']>1:
#             #     break
#             same.append(i['startIndex'])
#             text.append(i['startIndex'])
#         res.append(same)
#     if text:
#         a = set(source)
#         b = set(text)
#         temp = a & b
#         if len(temp) > 1:
#             for t in res:
#                 if set(source) & set(t) == set(t):
#                     source.remove(t[-1])
#     return source
# sentence = 'Bush, however, says he sees no short-term hope for a diplomatic solution to the gulf crisis at least until economic sanctions force Saddam to withdraw his army.'
# value = [0, 1, 6]
# t = corefs(sentence, value)
# print(t)
# nlp.close()


# result =  nlp.annotate( sentence, properties=
#                     {
#                         'timeout': '10000000',
#                         'annotators': 'coref',
#                         'outputFormat': 'json'
#                     })
#
# result = json.loads(result)
# # s = 'In its Friday issue, The Hamilton (Ontario) Spectator quoted an officer in the Amherst.'
# # nlp = spacy.load('en_coref_md')
# # doc = nlp(s)
# sen = 'She estimates her properties, worth a hundred thirty million dollars in October, are worth only half that now.'
# print(nlp.pos_tag(sen))
# from xml.dom.minidom import parse
#
# doc=parse("bioscope/abstracts.xml")                   #先把xml文件加载进来
# root=doc.documentElement                #获取元素的根节点
# document=root.getElementsByTagName('Document')
#
# for d in document:
#     cue = d.getElementsByTagName("cue")
#     for j in cue:
#         print(j.getAttribute('type'))
#         # print(j.childNodes[0].data)




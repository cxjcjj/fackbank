#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/11/2 21:28
from deal import pre_deal as pre
import Max
from Maxent import MaxEntropy
import eventden
import pickle
import sip as cnnatt
import cue as cnnlstm
from stanfordcorenlp import StanfordCoreNLP
import collections
from deal import data
from deal import parse
import numpy as np


import json
nlp = StanfordCoreNLP(r'E:\nlpData\stanford-corenlp-full-2018-02-27')

def get_file():
    sen, no_sen, sip, value = pre.get_data(r"factbank1.0/data/annotation//", 'event.txt', 'noevent.txt')
    pre.devide_file(sen, no_sen, sip, value, r"data//read//", '')

def max(event):
    sum_p = 0
    sum_r = 0
    sum_f = 0
    res = {'1': [0, 0, 0], '2': [0, 0, 0], '3': [0, 0, 0]}
    for i in range(10):
        # m = Max.MaxEnt()
        m = MaxEntropy()
        if event:
            print("event"+str(i))
            m.load_data(i)
            m.train()
            print(m.predictfile('data/result//compare_event_result_' + str(i + 1) + '.txt',
                            'data/result//event_result_' + str(i + 1) + '.txt'))
            resp, resr, resf, p, r, f = m.score()
            sum_p += p
            sum_r += r
            sum_f += f
        else:
            print("cue" + str(i))
            m.load_data(i, read_dir='data/read/', file='cue_')
            m.train()
            print(m.predictfile('data/result//compare_cue_result_' + str(i + 1) + '.txt','data/result//cue_result_' + str(i + 1) + '.txt'))
            resp, resr, resf, p, r, f = m.score()
            sum_p += p
            sum_r += r
            sum_f += f
            for key in res.keys():
                res[key][0] += resp[key]
                res[key][1] += resr[key]
                res[key][2] += resf[key]
    if not event:
        for key,value in res.items():
            print(key+':\tP:', value[0] / 10, "\tR:", value[1] / 10, '\tF', value[2]/ 10)
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)


def event_task():
    sum_p = 0
    sum_r = 0
    sum_f = 0
    ls = eventden.Den()
    ls.data()
    train = False
    for i in range(10):
        print("event" + str(i))
        ls.load_data(i)
        ls.build_model()
        if train:
            ls.train()
        ls.predict()
        ls.result_file('data/read//event_' + str(i + 1) + '.txt',
                       'data/result//compare_event_result_' + str(i + 1) + '.txt',
                       'data/result//event_result_' + str(i + 1) + '.txt')
        p, r, f = ls.score()
        sum_p += p
        sum_r += r
        sum_f += f
    print('sum\tP:%.4f\tR:%.4f\tF:%.4f' % (sum_p / 10, sum_r / 10, sum_f / 10))


def sip_task():
    sum_p = 0
    sum_r = 0
    sum_f = 0
    cnn = cnnatt.CNNatt()
    cnn.data()
    train = False
    for i in range(10):
        cnn.load_data(i)
        cnn.bulid_model()
        if train:
            cnn.train()
        cnn.predict()
        print('sip' + str(i))
        p, r, f = cnn.score()
        with open('data/out/sip_feature' + '.pkl', 'rb') as file:
            file_feature = pickle.load(file)
        cnn.result_file(file_feature)
        sum_p += p
        sum_r += r
        sum_f += f
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)
    print('sum\tP:%.4f\tR:%.4f\tF:%.4f' % (sum_p / 10, sum_r / 10, sum_f / 10))


def cue_task():
    sum_p = 0
    sum_r = 0
    sum_f = 0
    # ps:1  pr:2    neg:3
    res = {'1': [0, 0, 0], '2': [0, 0, 0], '3': [0, 0, 0]}
    ls = cnnlstm.cueModel()
    ls.data()
    train = False
    for i in range(10):
        print("cue" + str(i))
        ls.load_data(i)
        ls.build_model()
        if train:
            ls.train()
        ls.predict()
        print(ls.result_file('data/read//cue_' + str(i + 1) + '.txt',
                             'data/result//compare_cue_result_' + str(i + 1) + '.txt',
                             'data/result//cue_result_' + str(i + 1) + '.txt'))
        resp, resr, resf, p, r, f = ls.score()
        sum_p += p
        sum_r += r
        sum_f += f
        for key in res.keys():
            res[key][0] += resp[key]
            res[key][1] += resr[key]
            res[key][2] += resf[key]

    for key, value in res.items():
        print(key + ':\tP:%.4f\tR:%.4f\tF:%.4f' % (value[0] / 10, value[1] / 10, value[2] / 10))
    print('sum\tP:%.4f\tR:%.4f\tF:%.4f' % (sum_p / 10, sum_r / 10, sum_f / 10))


def get_sen_event(sips):
    sen_event = {}  # {sen_id:{event_id: sip}}
    sen_id = 0
    word_id = 0
    event_sip = collections.OrderedDict()
    for word in sips:
        if len(word)>= 2:
            words = word.strip().split('&')
            event = words[1]
            sip = words[2]
            if event == '1':
                event_sip[word_id] = sip
            word_id += 1
        else:
            sen_event[sen_id] = event_sip
            sen_id += 1
            word_id = 0
            event_sip = collections.OrderedDict()

    return sen_event


def corefs(sentence):
    output = nlp.annotate(sentence, properties={'annotators':'coref','outputFormat':'json','ner.useSUTime':'false'})
    out = json.loads(output)
    res = []
    text  = []
    for key, value in out['corefs'].items():
        # print(value)
        same = []
        for i in value:
            # 因为source长度都为1
            # if i['endIndex']- i['startIndex']>1:
            #     break
            same.append(i['endIndex']-1)
            text.append(i['endIndex']-1)
        res.append(same)
    return res, text


def GetRealSource(sen, sen_event):
    """
    对一个句子进行处理， 获取其中的相关源
    :param sen: str
    :param sen_event: {event_id:sip},
    :return: event_source: {event_id,:[source,..,source]}# 0 表示默认AUTHOR 1,2...表示在句子中的位置
    """
    event_source = {}
    token = nlp.word_tokenize(sen)
    sen_dep = nlp.dependency_parse(sen)
    pos = nlp.pos_tag(sen)
    r, leaf = parse.buildDenTree(sen_dep, token)
    res, text = corefs(sen)


    def get_source(dep, root, rs=[-2]):
        currs = rs.copy()
        k = root.id
        if root.id in sen_event.keys():
            event_source[root.id] = currs.copy()
            if sen_event[root.id] == '1':
                ns = -2
                pid = -2
                genflag = False
                # 获取引入源
                # 子结点中寻找ns
                for d in dep:
                    if d[2] == root.id+1:
                        pid = d[1]
                    if d[1] == root.id+1:
                        if d[0] == 'nsubj':
                            ns = d[2]
                        elif d[0] == 'nsubjpass':
                            ns = -1
                        elif 'pass' in d[0]:
                            genflag = True
                # 同阶关系中寻找ns
                if ns==-2 and pid != -2:
                    for d in dep:
                        if d[1] == pid:
                            if d[0] == 'nsubj':
                                ns = d[2]
                            elif d[0] == 'nsubjpass':
                                ns = -1
                            elif 'pass' in d[0]:
                                genflag = True
                if ns == -2 and genflag :
                    ns = -1
                flag = True
                # 判断是否和之前的词有同指

                for i in range(len(res)):
                    if ns in res[i]:
                        for j in res[i]:
                            if j in rs:
                                flag = False
                if flag and ns!= -2:
                    currs.append(ns)
        for ch in root.child:
            get_source(dep, ch, currs.copy())
    if ((token[0]=='\"' or token[0]=='``' ) and (len(token)<10 or token[-1]=='\"'or token[-1]=='\'\'')) or \
            ((len(token)<10 or token[0]=='\"' or token[0]=='``' ) and (token[-1]=='\"'or token[-1]=='\'\'')) :
        get_source(sen_dep,r, [-2, 0])
    else:
        get_source(sen_dep, r)
    sour = {-2:'AUTHOR', -1:'GEN',0: 'DUMMY' }
    for event, source in event_source.items():
        # temp = sorted(list(set(source)), reverse=True)
        source.reverse()
        sou = []
        for t in source:
            if t <=0:
                if sour[t] not in sou:
                    sou.append(sour[t])
            else:
                if token[t-1] not in sou:
                    sou.append(token[t-1])
        event_source[event] = sou

    return event_source


def source_task(sip_filename='sip_result_',out_dir=None, sen_dir=None, sip_result_dir=None):
    for i in range(10):
        print('source'+str(i))
        sen_file = sen_dir + '/sen_' + str(i + 1) + '.txt'
        sip_file = sip_result_dir + sip_filename + str(i + 1) + '.txt'
        com_file = out_dir + '/compare_source_result_' + str(i + 1) + '.txt'
        out_file = out_dir + '/source_result_' + str(i + 1) + '.txt'
        sens = open(sen_file, 'r', encoding='utf-8')
        sen_lines = sens.readlines()
        sips = open(sip_file, 'r', encoding='utf-8')
        sip_lines = sips.readlines()
        lines = sip_lines.copy()
        out = open(out_file, 'w', encoding='utf-8')
        com = open(com_file, 'w', encoding='utf-8')
        sen_id = 0
        word_id = 0
        # get sen_eventid
        sen_event = get_sen_event(sip_lines)
        sen_source = GetRealSource(sen_lines[sen_id].strip(), sen_event[sen_id])
        for word in lines:
            if len(word) >= 2:
                out.write(word.strip().split('&')[0]+'&')
                com.write(word.strip().split('&')[0] + '\t'+word.strip().split('&')[3]+'\t')
                if word_id in sen_source.keys():
                    out.write('_'.join(sen_source[word_id]))
                    com.write('_'.join(sen_source[word_id]))
                else:
                    out.write('0')
                    com.write('0')

                out.write('\n')
                com.write('\n')
                word_id += 1
            else:
                word_id = 0
                sen_id += 1
                if sen_id < len(sen_lines):
                    sen_source = GetRealSource(sen_lines[sen_id].strip(), sen_event[sen_id])
                out.write('\n')
                com.write('\n')
        sens.close()
        sips.close()
        out.close()
        com.close()
        # 计算结果
    scoresource('data/result/compare_source_result_')


def comfile(dir):
    for i in range(10):
        event_file = dir + 'event_result_' + str(i + 1) + '.txt'
        sip_file = dir + 'sip_result_' + str(i + 1) + '.txt'
        source_file = dir + 'source_result_' + str(i + 1) + '.txt'
        cue_file= dir + 'cue_result_' + str(i + 1) + '.txt'

        outfile = dir + 'feature_'+ str(i+1)+'.txt'
        eventf = open(event_file, 'r', encoding='utf-8')
        sipf = open(sip_file, 'r', encoding='utf-8')
        sourcef = open(source_file, 'r', encoding='utf-8')
        cuef = open(cue_file, 'r', encoding='utf-8')
        out = open(outfile, 'w', encoding='utf-8')

        for event, sip, source, cue in zip(eventf.readlines(), sipf.readlines(), sourcef.readlines()), cuef.readlines():
            if len(event)>=2:
                out.write(event.strip()+'&'+sip.strip().split('&')[1]+'&'+source.strip().split('&')[1]+'&'+cue.strip().split('&')[1]+'\n')
            else:
                out.write('\n')

        eventf.close()
        sipf.close()
        sourcef.close()
        cuef.close()
        out.close()


def get_senfeature(sen, event_feat, sip, cue):
    res = {}  # {event:{source：[event,sip,source, cue, sip_path, rs_path, cue_path],[]}]
    cue_type = {'1':'PS', '2':'PR', '3':'NEG','4':'PS_NEG','5':'PR_NEG'}
    token = nlp.word_tokenize(sen)
    dep = nlp.dependency_parse(sen)
    r, leaf = parse.buildDenTree(dep, token)
    for event,source in event_feat:
        feat = []
        feat.append(token[event])
        if event in sip:
            tempsip = sip.remove(event)
        else:
            tempsip = sip
        if tempsip:
            min = 100
            for i in tempsip:
                if abs(i-event)< min:
                    min = abs(i-event)
                    key = i
            feat.append(token[key])
            s = parse.two_route(leaf, feat[1], feat[0])
            s_path = ' '.join(s)
        else:
            feat.append('null')
            s_path = 'null'

        sources = []
        sour_feat = {}

        for sou in reversed(source.split('_')):
            sources.append(sou)
            temp = feat.copy()
            temp_source = list(reversed(sources))
            temp.append('_'.join(temp_source))
            temp_sou = '_'.join(temp_source)
            r_path = ''
            if temp_source.remove('AUTHOR'):
                for j in range(len(temp_source)):
                    if j:
                        r = parse.two_route(leaf, temp_source[j-1], temp_source[j])
                    else:
                        r = parse.get_route(leaf, temp_source[j])
                    r_path += ' '.join(r)
            else:
                r = parse.get_route(leaf, feat[0])
                r_path += ' '.join(r)
            if cue:
                cue_feat = []
                for cue_id, type in cue.items():
                    t = temp.copy()
                    scue = token[cue_id]
                    c = parse.two_route(leaf, scue, feat[0])
                    c_path = ' '.join(c)
                    t.append(scue)
                    t.append(s_path)
                    t.append(r_path)
                    t.append(c_path)
                    t.append(event-cue_id)
                    t.append(cue_type[type])
                    cue_feat.append(t)
                sour_feat[temp_sou] = cue_feat

            else:
                temp.append('null')
                temp.append(s_path)
                temp.append(r_path)
                temp.append('null')
                temp.append('0')
                temp.append('null')
                sour_feat[temp_sou] = [temp]
        res[event] = sour_feat

    return res


def getGanfeatures(dir='data/result/'):
    cue_type = {'null': '0', 'PS': '1', 'PR': '2', 'NEG': '3', 'PS_NEG': '4', 'PR_NEG': '5'}
    with open('data/read/sen_value' + '.pkl', 'rb') as f:
        file_val = pickle.load(f)  #[sen_id:{event:{source:value}}]
    for i in range(10):
        sen_filename = 'data/read/sen_' + str(i + 1) + '.txt'
        feat_filename = dir+'feature_' + str(i + 1) + '.txt'
        sen_val = file_val[i]
        sen_file = open(sen_filename, 'r')
        feat_file = open(feat_filename, 'r')
        out = open('data/read/ganfeature_'+str(i+1)+'.txt', 'w')
        outsen = open('data/read/gansen_'+str(i+1)+'.txt', 'w')

        sens = sen_file.readlines()
        feats = feat_file.readlines()
        sen_id = 0
        word_id = 0
        event_feat = []
        sip = []
        cue = {}
        token = []
        for feat in feats:
            if len(feat)>=2:
                temp = list(filter(None,feat.strip().split('&')))
                token.append(temp[0])
                event_f = temp[1]
                sip_f = temp[2]
                source_f = temp[3]
                cue_f = temp[4]
                if sip_f == '1':
                    sip.append(word_id)
                if event_f == '1':
                    event_feat.append([word_id, source_f])
                if cue_f != '0':
                    cue[word_id] = cue_f
                word_id += 1
            elif sen_id < len(sens):
                event_feature = get_senfeature(sens[sen_id].strip(), event_feat, sip, cue)
                for event, source_feat in event_feature.items():
                    for source, cue_feat in source_feat.items():
                        values = sen_val[sen_id][token[event]][source]
                        if values not in ['Uu', 'CT+', 'CT-', 'PS+', 'PS-', 'PR+', 'PR-']:
                            values = 'other'
                        c = []
                        c_type = []
                        for t in cue_feat:
                            for res in t:
                                out.write(str(res)+'\n')
                            c.append(t[3])
                            c_type.append(cue_type[t[-1]])
                        outsen.write(str(sen_id)+'\t'+sens[sen_id].strip() + '\t' + t[0] + '\t' + source + '\t' + '_'.join(c)+'\t'+ '_'.join(c_type)+'\t'+ values+'\n')
                        out.write('\n')
                event_feat = []
                sip = []
                cue = {}
                word_id = 0
                sen_id += 1
                token = []
        sen_file.close()
        feat_file.close()
        out.close()
        outsen.close()


def scoresource(file):
    sum_p = 0
    sum_r = 0
    sum_f = 0
    for i in range(10):
        filename = file + str(i + 1) + '.txt'
        f = open(filename, 'r', encoding='utf-8')
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for line in f.readlines():
            if len(line)>=2:
                token = line.strip().split()
                gold = token[1]
                pre = token[2]
                if gold != '0':
                    if gold != 'AUTHOR':
                        if pre == gold:
                            TP += 1
                        else:
                            FN += 1
                    else:
                        if pre != 'AUTHOR':
                            FP += 1
                        else:
                            TN += 1
        f.close()
        acc = (TP + TN)*1.0 /(TP+TN+FP+FN)


        if TP+FP == 0:
            P = 0.0
        else:
            P = TP*1.0/(TP+FP)
        if TP+FN == 0:
            R = 0
        else:
            R = TP*1.0/(TP+FN)
        if P == 0.0 or R == 0.0:
            F = 0.0
        else:
            F = 2 * P*R/(P+R)
        print('acc:', acc)
        print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
        print('P:', P, "\tR:", R, '\tF',F)
        sum_p += P
        sum_r += R
        sum_f += F
    print('ave\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)
    return sum_p, sum_r, sum_f

def score(file):
    sum_p = 0
    sum_r = 0
    sum_f = 0
    for i in range(10):
        filename = file + str(i + 1) + '.txt'
        f = open(filename, 'r', encoding='utf-8')
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for line in f.readlines():
            if len(line)>=2:
                token = line.strip().split()
                gold = token[1]
                pre = token[2]
                if gold != '0':
                    if pre == gold:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if pre != '0':
                        FP += 1
                    else:
                        TN += 1
        f.close()
        acc = (TP + TN)*1.0 /(TP+TN+FP+FN)


        if TP+FP == 0:
            P = 0.0
        else:
            P = TP*1.0/(TP+FP)
        if TP+FN == 0:
            R = 0
        else:
            R = TP*1.0/(TP+FN)
        if P == 0.0 or R == 0.0:
            F = 0.0
        else:
            F = 2 * P*R/(P+R)
        print('acc:', acc)
        print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
        print('P:', P, "\tR:", R, '\tF',F)
        sum_p += P
        sum_r += R
        sum_f += F
    print('ave\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)
    return sum_p, sum_r, sum_f

def judge_value(result, cues):
    res = []
    if result == 'Uu' or result == 'CT+':
        for cue in cues:
            res.append(0)
    elif result == 'CT-':
        for cue in cues:
            if cue == '3':
                res.append(1)
            else:
                res.append(0)
    elif result == 'PR+':
        for cue in cues:
            if cue == '2':
                res.append(1)
            else:
                res.append(0)
    elif result == 'PS+':
        for cue in cues:
            if cue == '1':
                res.append(1)
            else:
                res.append(0)
    elif result == 'PR-':
        for cue in cues:
            if cue == '2' or cue == '3' or cue == '5':
                res.append(1)
            else:
                res.append(0)
    elif result == 'PS-':
        for cue in cues:
            if cue == '1' or cue == '3' or cue == '4':
                res.append(1)
            else:
                res.append(0)
    else:
        for cue in cues:
            res.append(0)
    return res


def get_inputdata(featurefilename, labelfilename):
    """
    获取最后模型的输入
    :param featurefilename: 特征index文件
    :param labelfilename:  对应结果文件
    :return:
    """

    file_feature = []   # [sen_id, event, source, value]
    file_sample = []    # [[feature, label]]
    for i in range(10):
        sample = []
        fea = []
        featurefile = open(featurefilename+ str(i + 1) + '.txt', 'r', encoding='utf-8')
        labelfile = open(labelfilename+ str(i + 1) + '.txt', 'r', encoding='utf-8')
        features = featurefile.readlines()
        labels = labelfile.readlines()
        index = 0  # gansen
        feature_id = 1
        sen = {}
        feat = []
        j = 0
        for feature in features:
            if len(feature) >= 2:
                feat.append(feature.strip().split())
                if not feature_id % 9:
                    sample.append([feat])
                    feat = []
                feature_id += 1
            else:
                senfeature = labels[index].strip().split()
                sen_id = senfeature[0]
                event = senfeature[-5]
                source = senfeature[-4]
                cuetype = senfeature[-2].split('_')
                value = senfeature[-1]
                if value == 'Uu':
                    labelu = 0
                elif value in ['CT+', 'CT-', 'PS+', 'PS-', 'PR+', 'PR-']:
                    labelu = 1
                else:
                    labelu = 2
                ## judge PR PS 要写
                fea.append([sen_id, event, source, value, cuetype])
                rescue = judge_value(value, cuetype)
                if len(rescue) != len(cuetype):
                    print('code error')
                for t in rescue:
                    labelcue = t
                    label = []
                    label.append(labelu)
                    label.append(labelcue)
                    sample[j].append(label)
                    j += 1
                index += 1
        file_sample.append(sample)
        file_feature.append(fea)
    with open('sample.pkl', 'wb') as f:
        pickle.dump(file_sample, f, pickle.HIGHEST_PROTOCOL)
        print('dump file sample!')
    with open('find.pkl', 'wb') as f:
        pickle.dump(file_feature, f, pickle.HIGHEST_PROTOCOL)
        print('dump file find!')
    return


if __name__ == '__main__':
    # 处理好训练文件
    # get_file()
    # max(True)# 最大熵
    # score('data/result/compare_event_result_')
    # event_task()
    # sip_task()

    source_task('feature_','data/result/', 'data/read/', 'data/read/')  # train
    cue_task()
    # max(False)  # 最大熵
    comfile('data/result/')
    # getGanfeatures()  # test
    # getGanfeatures('data/read/')  #train
    # datas = data.Data()
    # datas.get_index_file('data/read/ganfeature_')
    # get_inputdata('data/read/feature_index', 'data/read/gansen_')




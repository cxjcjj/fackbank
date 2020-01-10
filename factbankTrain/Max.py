#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/11/10 20:21
from collections import defaultdict
from deal.data import Data
import math
import pickle

class MaxEnt(object):
    def __init__(self):
        self.features = defaultdict(int)
        self.features_Id = {}
        self.trainset = []
        self.testable = []
        self.test = []
        self.labels = set()
        self.result = []
        self.N = 0  # 样本数量
        self.n = 0  # 特征对(xi,yi)总数量
        self.feature_num = 0  # 特征数量
        self.weight = []  # 对应特征的权值
        self.sample_ep = []  # 样本分布的特征期望值
        self.model_ep = []  # 模型分布的特征期望值
        self.last_weight = []  # 上一轮迭代的权值
        self.EPS = 0.01  # 判断是否收敛的阈值
        self.split_word = '&'
        self.split_feature = '*'
        self.iter = 2
    def judge(self, word):
        # 表示不使用
        if word == '1':
            return True
        if not word:
            print('ww')
            return False
        if word in ['NN', 'JJ', 'JJS', 'NNS', 'VBN', 'VB', 'VBZ', 'VBP', 'JJR', 'NNP', 'IN', 'VBG', 'VBD']:
            return True
        else:
            return False

    def load_data(self, test_id, read_dir='data/read/', file= 'event_'):
        a = []
        for i in range(10):
            if i == test_id:
                flag = True
            else:
                flag = False
            filename = read_dir+file+str(i+1)+'.txt'
            file_sen = open(filename, 'r', encoding='utf-8')
            for sentence in file_sen.readlines():
                if len(sentence)>2:
                    sample = sentence.strip().split(self.split_word)
                    y = sample[-1]
                    X = sample[:-1]
                    X = list(filter(None, X))
                    # 事件
                    if file == 'event_':
                        pos = X[-2].split(self.split_feature)[2]
                    # 不确定和否定
                    else:
                        pos = '1'
                    # if y == '1':
                    #     # print(sample[0])
                    #     if not (pos in ['NN', 'JJ', 'JJS', 'NNS', 'VBN', 'VB', 'VBZ', 'VBP', 'JJR', 'NNP', 'IN', 'VBG', 'VBD']):
                    #         print(sentence)
                    #         print(sample[0] +';'+pos)
                    #     a.append(pos)
                    choose = self.judge(pos)
                    if X[0] in ['"','.','\\',',']:
                        choose = False
                    if flag:
                        self.test.append(sample)
                        if choose:
                            self.testable.append(1)
                        else:
                            self.testable.append(0)
                    else:
                        if choose:
                            self.trainset.append(sample)
                            self.labels.add(y)
                            for x in set(X):
                                self.features[(x,y)] += 1
                else:
                    if flag:
                        self.testable.append(0)
                        self.test.append('')
        print(set(a))

    def _initparams(self):
        self.N = len(self.trainset)
        # M param for GIS training algorithm
        self.n = len(self.features)
        self.feature_num = max([len(sample) - 1 for sample in self.trainset])
        self.sample_ep = [0.0] * self.n
        for i, xy in enumerate(self.features):
            self.sample_ep[i] = self.features[xy] * 1.0 / self.N
            self.features_Id[xy] = i
        self.weight = [0.0] * self.n
        self.last_weight = self.weight

    def _convergence(self, lastw, w):
        for w1, w2 in zip(lastw, w):
            print(abs(w1 - w2))
            if abs(w1 - w2) >= self.EPS:
                return False
        return True

    def probwgt(self, features, label):
        wgt = 0.0
        for f in features:
            if (f, label) in self.features:
                wgt += self.weight[self.features_Id[(f, label)]]
        return math.exp(wgt)

    """
    calculate feature expectation on model distribution
    """
    def Ep(self):
        ep = [0.0] * self.n
        for sample in self.trainset:
            features = sample[:-1]
            # calculate p(y|x)
            prob = self.calprob(features)
            for f in features:
                for w, l in prob:
                    # only focus on features from training data.
                    if (f, l) in self.features:
                        # get feature id
                        idx = self.features_Id[(f, l)]
                        # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                        ep[idx] += w * (1.0 / self.N)
        return ep

    def train(self):
        self._initparams()
        for i in range(self.iter):
            print('iter %d ...' % (i + 1))
            # calculate feature expectation on model distribution
            self.model_ep = self.Ep()
            self.last_weight = self.weight[:]
            for i, w in enumerate(self.weight):
                delta = 1.0 / self.feature_num * math.log(self.sample_ep[i] / self.model_ep[i])
                # update w
                self.weight[i] += delta
            # print(self.weight)
            # test if the algorithm is convergence
            if self._convergence(self.last_weight, self.weight):
                break

    def calprob(self, features):
        wgts = [(self.probwgt(features, l), l) for l in self.labels]
        Z = sum([w for w, l in wgts])
        prob = [(w / Z, l) for w, l in wgts]
        return prob

    def predict(self, comfile, resultfile):
        outfile = open(comfile, 'w', encoding="utf-8")
        outfeature = open(resultfile, 'w', encoding="utf-8")

        for sample, flag in zip(self.test, self.testable):
            if sample:
                features = sample[:-1]
                label = sample[-1]
                if flag:
                    prob = self.calprob(features)
                    result = sorted(prob, key=lambda x: (x[0], x[1]), reverse=True)
                    outfeature.write(features[0]+self.split_word+result[0][1]+'\n')
                    outfile.write(features[0] + '\t' + label + '\t' + result[0][1] + '\n')
                    self.result.append([features[0], label, result[0][1]])
                else:
                    outfeature.write(features[0]+self.split_word+'0\n')
                    outfile.write(features[0] + '\t' + label + '\t' + '0\n')
            else:
                outfeature.write('\n')
                outfile.write('\n')
        outfile.close()
        outfeature

    def saveModel(self):
        '''
        将中间数据存入模型文件
        :param modelFile:
        :return:
        '''

        with open('model.pickle', 'wb') as f:
            pickle.dump(
                [self.trainset, self.labels, self.features, self.N, self.n,  self.feature_num, self.sample_ep, self.model_ep,
                 self.weight, self.last_weight, self.EPS], f)
        f.close()

    def loadModel(self):
        '''
        加载模型文件
        :return:
        '''
        with open('model.pickle', 'rb') as f:
            self.trainset, self.labels, self.features, self.N, self.n, self.feature_num, self.sample_ep, self.model_ep,
            self.weight, self.last_weight, self.EPS = pickle.load(
                f)
        # print(self._samples, self._Y, self._numXY, self._N, self._n, self._xyID, self._C, self._ep_, self._ep,
        #       self._w, self._lastw, self._EPS)
        f.close()

    def score(self, flag=True):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        label = list(sorted(list(self.labels)))
        n = len(label)
        TP_C = [0] * n
        FP_C = [0] * n
        TN_C = [0] * n
        FN_C = [0] * n
        P_C = [0] * n
        R_C = [0] * n
        F_C =[0] * n
        g = [r[1] for r in self.result]
        p = [r[2] for r in self.result]
        # print('sk\tf1:'+ str(f1_score(g,p,average='macro')))
        # cal_F1_measure(g, p)
        for i in range(n):
            for token, gold, pre in self.result:
                if pre == label[i]:
                    pre = 1
                else:
                    pre = 0
                if gold == label[i]:
                    gold = 1
                else:
                    gold = 0
                if gold != 0:
                    if pre == gold:
                        TP_C[i] += 1
                    else:
                        FN_C[i] += 1
                else:
                    if pre != 0:
                        FP_C[i] += 1
                    else:
                        TN_C[i] += 1
        for token, gold, pre in self.result:
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

        macroP = 0
        macroR = 0
        macroF = 0
        iTP = 0
        iFN = 0
        iFP = 0
        resultP = {}
        resultR = {}
        resultF = {}
        for i in range(n):
            P_C[i] = TP_C[i]*1.0/(TP_C[i]+FP_C[i])
            R_C[i] = TP_C[i]*1.0/(TP_C[i]+FN_C[i])
            F_C[i] = 2 * P_C[i]*R_C[i]/(P_C[i]+R_C[i])
            print('class'+label[i]+'\tTP:', TP_C[i], '\tTN:', TN, '\tFP', FP_C[i], '\tFN', FN)
            print('\t'+'P:', P_C[i], "\tR:", R_C[i], '\tF', F_C[i])
            resultP[label[i]] = P_C[i]
            resultR[label[i]] = R_C[i]
            resultF[label[i]] = F_C[i]
            iTP += TP_C[i]
            iFP += FP_C[i]
            iFN += FN_C[i]
            macroP += P_C[i]
            macroR += R_C[i]
            macroF += F_C[i]
        if n > 2:
            microP = (iTP * 1.0)/(iTP+iFP) # n被约掉
            microR = (iTP * 1.0)/(iTP+iFN)
            microF = 2*microP*microR / (microP+microR)
            macroP = macroP / n
            macroR = macroR / n
            macroF = macroF / n
            print('\t' + 'macroP:', macroP, "\tmacroR:", macroR, '\tmacroF', macroF)
            print('\t' + 'microP:', microP, "\tmicroR:", microR, '\tmicroF', microF)
            # resultP['macro'] = macroP
            # resultP['micro'] = microP
            # resultR['macro'] = macroR
            # resultR['micro'] = microR
            # resultF['macro'] = macroF
            # resultF['micro'] = microF

        acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        P = TP*1.0/(TP+FP)
        R = TP*1.0/(TP+FN)
        F = 2 * P*R/(P+R)
        if flag:
            print('acc:', acc)
            print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
        print('P:', P, "\tR:", R, '\tF',F)
        return resultP,resultR,resultF, P, R, F


if __name__ == '__main__':
    sum_p = 0
    sum_r = 0
    sum_f = 0
    res = {'1': [0, 0, 0], '2': [0, 0, 0], '3': [0, 0, 0]}
    for i in range(10):
        m = MaxEnt()
        # m.load_data(i)
        m.load_data(i, read_dir='data/read/', file= 'cue_')
        m.train()
        # print(m.predict('data/result//compare_event_result_'+str(i+1)+'.txt', 'data/result//event_result_'+str(i+1)+'.txt'))
        print(m.predict('data/result//compare_cue_result_' + str(i + 1) + '.txt','data/result//cue_result_' + str(i + 1) + '.txt'))
        resp, resr, resf, p, r, f = m.score()
        sum_p += p
        sum_r += r
        sum_f += f
        for key in res.keys():
            res[key][0] += resp[key]
            res[key][1] += resr[key]
            res[key][2] += resf[key]

    for key, value in res.items():
        print(key + ':\tP:', value[0] / 10, "\tR:", value[1] / 10, '\tF', value[2] / 10)
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)
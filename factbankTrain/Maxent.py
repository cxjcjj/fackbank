#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/12/4 16:39
import collections
import math


class MaxEntropy():
    def __init__(self):
        self.trainset = []  # 样本集，元素是[y,x1,x2,...]的样本
        self.testable = []
        self.test = []
        self._Y = set([])  # 标签集合，相当去去重后的y
        self._numXY = collections.defaultdict(int)  # key为(x,y)，value为出现次数
        self._N = 0  # 样本数
        self._ep_ = []  # 样本分布的特征期望值
        self._xyID = {}  # key记录(x,y),value记录id号
        self._n = 0  # 特征的个数
        self._C = 0  # 最大特征数
        self._IDxy = {}  # key为(x,y)，value为对应的id号
        self._w = []
        self._EPS = 0.005  # 收敛条件
        self._lastw = []  # 上一次w参数值
        self.split_word = '&'
        self.split_feature = '*'
        self.result = []

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
        for i in range(10):
            if i == test_id:
                flag = True
            else:
                flag = False
            filename = read_dir + file + str(i + 1) + '.txt'
            file_sen = open(filename, 'r', encoding='utf-8')
            for sentence in file_sen.readlines():
                if len(sentence) > 2:
                    sample = sentence.strip().split(self.split_word)
                    y = sample[-1]
                    X = sample[:-1]
                    X = list(filter(None, X))
                    self._Y.add(y)
                    # 事件
                    if file == 'event_':
                        # pos = X[-2].split(self.split_feature)[2]
                        pos = '1'
                    # 不确定和否定
                    else:
                        pos = '1'

                    choose = self.judge(pos)
                    if X[0] in ['"', '.', '\\', ',']:
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
                            self._Y.add(y)
                            for x in set(X):
                                self._numXY[(x, y)] += 1
                else:
                    if flag:
                        self.testable.append(0)
                        self.test.append('')

    def _sample_ep(self):  # 计算特征函数fi关于经验分布的期望
        self._ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def _initparams(self):  # 初始化参数
        self._N = len(self.trainset)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in self.trainset])
        self._w = [0] * self._n
        self._lastw = self._w[:]

        self._sample_ep()  # 计算每个特征关于经验分布的期望

    def _Zx(self, X):  # 计算每个x的Z值
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):  # 计算每个P(y|x)
        Z = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss) / Z
        return pyx

    def _model_ep(self, index):  # 计算特征函数fi关于模型的期望
        x, y = self._IDxy[index]
        ep = 0
        for sample in self.trainset:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def _convergence(self):
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._EPS:
                return False
        return True

    def predict(self, X):  # 计算预测概率
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result

    def predictfile(self, comfile, resultfile):
        outfile = open(comfile, 'w', encoding="utf-8")
        outfeature = open(resultfile, 'w', encoding="utf-8")

        for sample, flag in zip(self.test, self.testable):
            if sample:
                features = sample[:-1]
                label = sample[-1]
                if flag:
                    prob = self.predict(features)
                    result = sorted(prob.items(), key=lambda x: x[1], reverse=True)
                    outfeature.write(features[0]+self.split_word+result[0][0]+'\n')
                    outfile.write(features[0] + '\t' + label + '\t' + result[0][0] + '\n')
                    self.result.append([features[0], label, result[0][0]])
                else:
                    outfeature.write(features[0]+self.split_word+'0\n')
                    outfile.write(features[0] + '\t' + label + '\t' + '0\n')
            else:
                outfeature.write('\n')
                outfile.write('\n')
        outfile.close()
        outfeature

    def train(self, maxiter=1):  # 训练数据
        self._initparams()
        for loop in range(0, maxiter):  # 最大训练次数
            print("iter:%d" % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)  # 计算第i个特征的模型期望
                self._w[i] += math.log(self._ep_[i] / ep) / self._C  # 更新参数
            if self._convergence():  # 判断是否收敛
                break

    def score(self, flag=True):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        label = list(sorted(list(self._Y)))
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
    for i in range(10):
        m = MaxEntropy()
        m.load_data(i)
        # m.load_data(i, read_dir='data/read/', file= 'cue_')
        m.train()
        print(m.predictfile('data/result//compare_event_result_'+str(i+1)+'.txt', 'data/result//event_result_'+str(i+1)+'.txt'))
        # print(m.predict('data/result//compare_cue_result_' + str(i + 1) + '.txt','data/result//cue_result_' + str(i + 1) + '.txt'))
        resp,resr,resf,p,r,f = m.score()
        sum_p += p
        sum_r += r
        sum_f += f

    print('sum\tP:', sum_p/10, "\tR:", sum_r/10, '\tF', sum_f/10)

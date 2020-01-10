#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/12/26 15:40

from deal import data
import pickle
from deal.data import Data
import keras.backend as K
from keras.layers import Input, Dense, TimeDistributed, Reshape, Flatten, Embedding, Dropout, LSTM, Bidirectional, \
    concatenate, Multiply, Dot, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D, MaxPooling3D, MaxPooling1D, \
    multiply
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import Sequential, Model, load_model
from keras.layers.core import Lambda
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical


class cueModel:
    def __init__(self):
        self.event_num = 0
        self.dim = 100
        self.label = {}
        self.label_size = 2
        self.feature_size = 3
        self.maxlen = 1
        self.train_len = 1
        self.sen_size = 200
        self.train_epochs = 60
        self.batch = 100
        self.display_step = 100
        self.learn_rate = 0.001
        self.saver_dir = 'data/result'
        self.train_x = []
        self.train_y = []
        self.train_f = []
        self.train_l = []
        self.valid_x = []
        self.valid_y = []
        self.valid_f = []
        self.test_x = []
        self.test_y = []
        self.test_f = []
        self.test_l = []
        self.gold = []
        self.result = []
        self.emb = None
        self.i = 0
        self.split_word = '&'
        self.split_feature = '*'
        self.best_model = None
        self.file = None

    def data(self):
        datas = data.Data()
        datas.pre_file = 'cue'
        datas.build_index()
        file_ins, file_ins_id, file_sen, file_sen_id = datas.read_instance()
        datas.build_embedding('glove.6B.100d.txt', datas.word_index)
        self.emb = datas.emb
        print(datas.label_index.instance_index)
        temp_label = datas.label_index.instance_index
        for key, value in temp_label.items():
            self.label[value] = key
        self.label_size = len(self.label)
        self.file = file_ins_id

    def load_data(self, index):
        k = 9 if index != 9 else 8
        maxlen = 0
        test_x = []
        train_x = []
        valid_x = []
        test_y = []
        train_y = []
        valid_y = []
        gold = []
        for i in range(10):
            ins = self.file[i]
            if i == index:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]

                    for w,f,l in zip(words, features, labels):
                        x = []
                        x.append(w)
                        for fea in f:
                            for tf in fea:
                                x.append(tf)
                        test_x.append(x)
                        test_y.append(l)
                        gold.append(l)

            elif i == k:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]

                    for w, f, l in zip(words, features, labels):
                        x = []
                        x.append(w)
                        for fea in f:
                            for tf in fea:
                                x.append(tf)
                        valid_x.append(x)
                        valid_y.append(l)
            else:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    for w, f, l in zip(words, features, labels):
                        x = []
                        x.append(w)
                        for fea in f:
                            for tf in fea:
                                x.append(tf)
                        train_x.append(x)
                        train_y.append(l)

        self.gold = gold
        self.test_x = test_x
        self.train_x = train_x
        self.valid_x = valid_x
        self.test_y = test_y
        self.train_y = train_y
        self.valid_y = valid_y
        self.train_len = len(self.test_x[0])

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.emb), output_dim=100, input_length=self.train_len, weights=[self.emb],
                            trainable=False))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.label_size, activation='softmax'))
        opt = SGD(lr=self.learn_rate)
        optimizer = RMSprop(lr=self.learn_rate)
        model.compile(optimizer='rmsprop',  # 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                      loss='sparse_categorical_crossentropy',  # 等价于loss = losses.binary_crossentropy
                      metrics=['accuracy'])  # 等价于metrics = [metircs.binary_accuracy]
        # model.summary()

        self.model = model
        return model

    def train(self):
        tensorboard = TensorBoard(log_dir='data/log')
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filename = save_dir + '/cue/cbest.hdf5'

        checkpoint = ModelCheckpoint(filepath=filename, monitor='val_accuracy', mode='auto', save_best_only='True')

        callback_lists = [tensorboard, checkpoint]

        train_x = np.array(self.train_x)
        valid_x = np.array(self.valid_x)

        train_y = self.train_y # np.eye(self.label_size)[self.train_y]
        valid_y = self.valid_y # np.eye(self.label_size)[self.valid_y]
        self.model.fit(train_x, train_y,
                       batch_size=self.batch,
                       epochs=self.train_epochs,
                       validation_data=(valid_x, valid_y),
                       shuffle=1,
                       verbose=1,
                       callbacks=callback_lists)

    def predict(self):
        test_x = np.array(self.test_x)
        name = 'saved_models/cue'
        best_model_name = sorted(os.listdir(name)).pop()
        self.best_model = load_model(os.path.join(name, best_model_name), custom_objects={'loss': 'sparse_categorical_crossentropy',
                                                     'accuracy': 'accuracy'}) # load_model(os.path.join(name, best_model_name), custom_objects={"CRF": CRF,"crf_loss": crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy})
        result = self.best_model.predict(test_x)
        self.result = result.argmax(axis=-1)
        # self.result = [1 if res >= 0.1 else 0 for res in result]
        self.score()

    def result_file(self, readfile, comfile, resultfile):
        outfile = open(comfile, 'w', encoding="utf-8")
        outresult = open(resultfile, 'w', encoding="utf-8")
        read = open(readfile, 'r', encoding='utf-8')
        lines = read.readlines()
        i = 0
        if len(self.result) != len(self.gold):
            print('size error')
        for line in lines:
            if len(line) > 1:
                token = line.strip().split(self.split_word)
                word = token[0]
                label = token[-1]
                outresult.write(word + self.split_word + self.label[self.result[i]] + '\n')
                outfile.write(word + '\t' + label + '\t' + self.label[self.result[i]] + '\n')
                i += 1
            else:
                outresult.write('\n')
                outfile.write('\n')
        outfile.close()
        outresult.close()

    def score(self):
        print(self.label)
        if len(self.gold) != len(self.result):
            print('label list length Error')
            return None
        label_counter = Counter(self.gold)
        error = Counter(self.result)
        # Counter(计数器)是对字典的补充，用于追踪值的出现次数
        # most_common(a)截取指定位数的值，截取前a位的值
        for label, num in label_counter.most_common():
            print(label, '-----', num)
            # label是正确标签的数值
        print("pre:")
        for label, num in error.most_common():
            print(label, '-----', num)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        n = self.label_size
        TP_C = [0] * n
        FP_C = [0] * n
        TN_C = [0] * n
        FN_C = [0] * n
        P_C = [0] * n
        R_C = [0] * n
        F_C = [0] * n
        i = 0
        label = []
        for key, value in self.label.items():
            label.append(value)
            for gold, pre in zip(self.gold, self.result):
                if pre == key :
                    pre = 1
                else:
                    pre = 0
                if gold == key:
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
            i += 1
        for gold, pre in zip(self.gold, self.result):
            if gold != 0:
                if pre == gold:
                    TP += 1
                else:
                    FN += 1
            else:
                if pre != 0:
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
            if (TP_C[i] + FP_C[i]) == 0:
                P_C[i] = 0
            else:
                P_C[i] = TP_C[i] * 1.0 / (TP_C[i] + FP_C[i])
            if (TP_C[i] + FN_C[i]) == 0:
                R_C[i] = 0
            else:
                R_C[i] = TP_C[i] * 1.0 / (TP_C[i] + FN_C[i])
            if P_C[i] + R_C[i] == 0:
                F_C[i] = 0
            else:
                F_C[i] = 2 * P_C[i] * R_C[i] / (P_C[i] + R_C[i])
            print('class' + label[i] + '\tTP:', TP_C[i], '\tTN:', TN, '\tFP', FP_C[i], '\tFN', FN)
            print('\t' + 'P:', P_C[i], "\tR:", R_C[i], '\tF', F_C[i])
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
            microP = (iTP * 1.0) / (iTP + iFP)  # n被约掉
            microR = (iTP * 1.0) / (iTP + iFN)
            microF = 2 * microP * microR / (microP + microR)
            macroP = macroP / n
            macroR = macroR / n
            macroF = macroF / n
            print('\t' + 'macroP:', macroP, "\tmacroR:", macroR, '\tmacroF', macroF)
            print('\t' + 'microP:', microP, "\tmicroR:", microR, '\tmicroF', microF)
            resultP['macro'] = macroP
            resultP['micro'] = microP
            resultR['macro'] = macroR
            resultR['micro'] = microR
            resultF['macro'] = macroF
            resultF['micro'] = microF
        print('TP:', TP, '\tTN:', TN, '\tFP', FP, '\tFN', FN)
        acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        if TP + FP == 0:
            P = 0
        else:
            P = TP * 1.0 / (TP + FP)
        if TP + FN ==0:
            R = 0
        else:
            R = TP * 1.0 / (TP + FN)
        if P+R == 0:
            F = 0
        else:
            F = 2 * P * R / (P + R)
        print('acc:', acc)
        print('P:', P, "\tR:", R, '\tF', F)
        return resultP, resultR, resultF, P, R, F


if __name__ == "__main__":
    sum_p = 0
    sum_r = 0
    sum_f = 0
    res = {'1': [0, 0, 0], '2': [0, 0, 0], '3': [0, 0, 0]}
    ls = cueModel()
    ls.data()
    for i in range(10):
        print("cue" + str(i))
        ls.load_data(i)
        ls.build_model()
        ls.train()
        ls.predict()
        print(ls.result_file('data/read//event_' + str(i + 1) + '.txt',
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
        print(key + ':\tP:', value[0] / 10, "\tR:", value[1] / 10, '\tF', value[2] / 10)
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)







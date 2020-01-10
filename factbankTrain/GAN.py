#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/11/27 10:19

from __future__ import print_function
from keras.utils import to_categorical
from collections import defaultdict
import pickle
from PIL import Image
from keras.utils import plot_model
from six.moves import range

import keras.backend as K
from keras.initializers import Constant
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, LSTM, Bidirectional, concatenate,Multiply, Dot
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.layers.core import Lambda
import numpy as np
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
import data
import os
from data import cal_F1_measure,cal_macro_micro_F1, scoreF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(1337)

class ACGAN:
    def __init__(self):
        self.epochs = 5
        self.batch = 100
        self.num_words = 4396
        self.maxlen = [20, 20, 20]
        self.lr = 0.01
        self.dropout = 0.5
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        self.embdding = None
        self.index = 9
        self.lenth = 20
        return

    def load_data(self, index):
        self.maxlen = data.produce('sample.pkl', 'mydata.pkl', index)
        self.train_set, self.valid_set,self.test_set = data.load('mydata.pkl')
        self.embdding = np.array(data.load('word_emb.pkl'))
        return

    def lstm_layer(self, x, maxlen):
        lstm = Sequential()
        lstm.add(Embedding(self.num_words, 100, input_length=maxlen))
        lstm.add(Dropout(self.dropout))
        lstm.add(Dense(maxlen))
        lstm.add(LSTM(100, return_sequences=True, dropout=self.dropout, activation='sigmoid'))
        lstm.add(Dense(100, activation='softmax'))
        return lstm(x)

    def build_generator(self):
        # this is the z space commonly refered to in GAN papers
        z0 = Input(shape=(self.maxlen[0],), dtype='float32')
        z1 = Input(shape=(self.maxlen[1],), dtype='float32')
        z2 = Input(shape=(self.maxlen[2],), dtype='float32')
        # this will be our label
        lu = Input(shape=(1,), dtype='float32')
        lcue = Input(shape=(1,), dtype='float32')

        def mul(x, y):
            return layers.multiply([x,y])
        temp = Lambda(mul, arguments={'y': lcue})(lu)
        x0 = Lambda(mul, arguments={'y': temp})(z0)
        x1 = Lambda(mul, arguments={'y': temp})(z1)
        x2 = Lambda(mul, arguments={'y': temp})(z2)
        fake0= self.lstm_layer(x0, self.maxlen[0])
        fake1 = self.lstm_layer(x1, self.maxlen[1])
        fake2 = self.lstm_layer(x2, self.maxlen[2])

        return Model(inputs=[z0, z1, z2, lu, lcue], outputs=[fake0, fake1, fake2])


    def bilstm_layer(self, x, maxlen):
        bilstm = Sequential()
        bilstm.add(Bidirectional(LSTM(50, input_shape=(maxlen, 100), return_sequences=True, dropout=self.dropout)))
        bilstm.add(Dense(maxlen))
        p = bilstm(x)
        m = Dense(1, activation='tanh')(p)
        a = Dense(1, activation='softmax')(m)
        def mult(x, y):
            return K.batch_dot(x, y, axes=1)
        h = Lambda(mult, arguments={'y': a})(p)
        out = Dense(1, activation='tanh')(h)

        return out


    def build_discriminator(self):
        sip = Input(shape=(self.maxlen[0], 100,), dtype='float32')
        rs = Input(shape=(self.maxlen[1], 100,), dtype='float32')
        cue = Input(shape=(self.maxlen[2], 100,), dtype='float32')
        i = Input(shape=(1, 100, ), dtype='int32')
        t = Input(shape=(1, 100, ), dtype='int32')
        hs = self.bilstm_layer(sip, self.maxlen[0])
        hr = self.bilstm_layer(rs, self.maxlen[1])
        hc = self.bilstm_layer(cue, self.maxlen[2])
        hs = Reshape((self.maxlen[0],))(hs)
        hr = Reshape((self.maxlen[1],))(hr)
        hc = Reshape((self.maxlen[2],))(hc)
        def intofloat(x, ty):
            return (K.cast(x, dtype=ty))
        li = Reshape((100,))(Lambda(intofloat, arguments={'ty': 'float32'})(i))
        lt = Reshape((100,))(Lambda(intofloat, arguments={'ty': 'float32'})(t))
        fu = concatenate([hs, hr])
        fcue = concatenate([li, lt, hc])
        outu = Dense(3, activation='softmax')(fu)
        outc = Dense(2, activation='softmax')(fcue)
        fakes = Dense(1, activation='softmax')(hs)
        faker = Dense(1, activation='softmax')(hr)
        fakec = Dense(1, activation='softmax')(hc)
        fakes = concatenate([fakes, faker, fakec])
        rfake = Dense(1, activation='sigmoid')(fakes)

        return Model(inputs=[sip, rs, cue, i, t], outputs=[rfake, outu, outc])


def emb(emb, x):
    y = np.zeros((x.shape[0],x.shape[1], 100))
    for k in range(len(x)):
        for i in range(len(x[k])):
            y[k][i] = emb[x[k][i]]
    return y


def emby(emb, x):
    y = np.zeros((x.shape[0], 100))
    for k in range(len(x)):
        y[k] = emb[x[k]]
    return y.reshape((x.shape[0], 1, 100))


def change(x, num):
    y = np.zeros((x.shape[0], num))
    for i in range(len(x)):
        y[i] = to_categorical(x[i], num)
    return y


def judge_label(u_label, cue_label, cue, value):
    dic = {0:'Uu', 1:'PS+', 2:'PR+',3:'CT+', -1:'PS-', -2:'PR-',-3:'CT-'}
    i = 0
    apply = []
    if '2' in u_label:
        return 'other'
    if len(set(u_label)) == 1 and 0 in set(u_label):
        return 'Uu'
    for lu, lc in zip(u_label, cue_label):
        if lu == 0:
            res = 0
        else:
            if lc == 1:
                apply.append(i)
        i += 1
    pre = 1
    v = 0
    va = 3
    if apply:
        for j in apply:
            if cue[j] == '3':
                pre = -1
            elif cue[j] == '1' or cue[j]=='2':
                if value[j] > v:
                    v = value[j]
                    va = int(cue[j])
        res = pre * va
    else:
        res = 3

    return dic[res]


def get_pre(resu, reacue, resvalue, index):
    #[sen_id, event, source, value, c]
    with open('find.pkl', 'rb') as file:
        file_feature = pickle.load(file)
    feature = file_feature[index]
    result_pre = []
    result_gold = []
    author_pre = []
    author_gold = []
    in_pre = []
    in_gold = []
    i = 0
    for feat in feature:
        prefeature = feat[:3]
        value = feat[-2]
        cue = feat[-1]
        sou = feat[2]
        result_gold.append(value)
        lenth = len(cue)
        pre = judge_label(resu[i:i+lenth], reacue[i:i+lenth], cue, resvalue[i:i+lenth])
        i = i + lenth
        result_pre.append(pre)
        if sou == 'AUTHOR':
            author_gold.append(value)
            author_pre.append(pre)
        else:
            in_gold.append(value)
            in_pre.append(pre)
    return result_gold, result_pre, author_gold, author_pre, in_gold, in_pre


def score(golds, pres, flag=True):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for gold, pre in zip(golds, pres):
        if gold != 'Uu':
            if pre == gold:
                TP += 1
            else:
                FN += 1
        else:
            if pre != 'Uu':
                FP += 1
            else:
                TN += 1
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
    if flag:
        print('acc:', acc)
        print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
    print('P:', P, "\tR:", R, '\tF',F)
    return P, R, F


def train_gan(key):
    ac = ACGAN()
    # get data
    ac.load_data(key)
    tr_x0, tr_x1, tr_x2, tr_x3, tr_x4, tr_x5, tr_x6, tr_x7, tr_x8, tr_y0, tr_y1, tr_y_one = ac.train_set
    va_x0, va_x1, va_x2, va_x3, va_x4, va_x5, va_x6, va_x7, va_x8, va_y0, va_y1, va_y_one = ac.valid_set
    te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1, te_y_one = ac.test_set

    # build discriminator
    discriminator = ac.build_discriminator()
    discriminator.compile(optimizer=Adam(lr=ac.lr, beta_1=0.5),
                          loss=['binary_crossentropy', 'sparse_categorical_crossentropy',
                                'sparse_categorical_crossentropy'])
    # build generator
    generator = ac.build_generator()
    generator.compile(optimizer=Adam(lr=ac.lr, beta_1=0.5),
                      loss='binary_crossentropy')

    j = 0
    f = 0
    for epoch in range(ac.epochs):
        print('Epoch {} of {}'.format(epoch + 1, ac.epochs))

        batch = int(len(ac.train_set[0]) / ac.batch)
        progress_bar = Progbar(target=batch)

        epoch_disc_loss = []
        for i in range(batch):
            # progress_bar.update(i)
            # generate a new batch of noise
            begin = i * ac.batch
            end = (i + 1) * ac.batch if (i + 1) * ac.batch < len(ac.train_set[0]) else len(ac.train_set[0])
            lenth = end - begin
            noise0 = np.random.random((lenth, ac.maxlen[0]))
            noise1 = np.random.random((lenth, ac.maxlen[1]))
            noise2 = np.random.random((lenth, ac.maxlen[2]))

            # get a batch of real daata
            t = np.array(tr_x4)
            temp_sip = np.array(tr_x4[begin: end])
            temp_rs = np.array(tr_x5[begin: end])
            temp_cue = np.array(tr_x6[begin: end])
            real_sip = emb(ac.embdding, temp_sip)
            real_rs = emb(ac.embdding, temp_rs)
            real_cue = emb(ac.embdding, temp_cue)
            idx = np.array(tr_x7[begin: end])
            ty = np.array(tr_x8[begin: end])
            index = emby(ac.embdding, idx)
            tp = emby(ac.embdding, ty)

            tr_yu = np.array(tr_y0[begin: end]).reshape((lenth, 1))
            tr_ycue = np.array(tr_y1[begin: end]).reshape((lenth, 1))
            tr_yf = np.array(tr_y_one[begin: end])

            fake_sip, fake_rs, fake_cue = generator.predict([noise0, noise1, noise2, tr_yu, tr_ycue])
            sips = np.concatenate([real_sip, fake_sip], axis=0)
            rss = np.concatenate([real_rs, fake_rs], axis=0)
            cues = np.concatenate([real_cue, fake_cue], axis=0)
            indexs = np.concatenate([index, index], axis=0)
            types = np.concatenate([tp, tp], axis=0)
            tr_f = np.concatenate([tr_yf, tr_yf]).reshape((-1, 1))
            tr_u = np.concatenate([tr_yu, tr_yu]).reshape((-1, 1))
            tr_cue = np.concatenate([tr_ycue, tr_ycue]).reshape((-1, 1))

            disc_loss = discriminator.train_on_batch([sips, rss, cues, indexs, types], [tr_f, tr_u, tr_cue])
            epoch_disc_loss.append(disc_loss)
        print('\nTesting for epoch {}:'.format(epoch + 1))

        # get valid data
        temp_sip = np.array(va_x4)
        temp_rs = np.array(va_x5)
        temp_cue = np.array(va_x6)
        real_sip = emb(ac.embdding, temp_sip)
        real_rs = emb(ac.embdding, temp_rs)
        real_cue = emb(ac.embdding, temp_cue)
        idx = np.array(va_x7)
        ty = np.array(va_x8)
        index = emby(ac.embdding, idx)
        tp = emby(ac.embdding, ty)

        va_yu = np.array(va_y0).reshape((-1, 1))
        va_ycue = np.array(va_y1).reshape((-1, 1))
        va_yf = np.array(va_y_one).reshape((-1, 1))
        size = va_ycue.shape[0]
        discriminator.train_on_batch([real_sip, real_rs, real_cue, index, tp], [va_yf, va_yu, va_ycue])

        # get test data
        temp_sip = np.array(te_x4)
        temp_rs = np.array(te_x5)
        temp_cue = np.array(te_x6)
        real_sip = emb(ac.embdding, temp_sip)
        real_rs = emb(ac.embdding, temp_rs)
        real_cue = emb(ac.embdding, temp_cue)
        idx = np.array(te_x7)
        ty = np.array(te_x8)
        index = emby(ac.embdding, idx)
        tp = emby(ac.embdding, ty)

        te_yu = np.array(te_y0).reshape((-1, 1))
        te_ycue = np.array(te_y1).reshape((-1, 1))
        te_yf = np.array(te_y_one).reshape((-1, 1))
        size = te_ycue.shape[0]
        yf, yu, ycue = discriminator.predict([real_sip, real_rs, real_cue, index, tp])
        goldu = te_yu.reshape((1, size))[0]
        preu = yu.argmax(axis=-1).reshape((1, size))[0]
        goldcue = te_yu.reshape((1, size))[0]
        precue = yu.argmax(axis=-1).reshape((1, size))[0]
        cuevalue = ycue.max(axis=-1).reshape((1, size))[0]
        # f1_resultu, f_sumu = cal_F1_measure(goldu, preu)
        # f1_resultcue, f_sumcue = cal_F1_measure(goldcue, precue)
        # f_sum = f_sumu +f_sumcue
        gold, pre, au_gold, au_pre, in_gold, in_pre = get_pre(preu, precue, cuevalue, key)
        P, R, F = score(gold, pre, False)
        # f1_result = cal_F1_measure(gold, pre)
        # micro_F1, macro_F1 = cal_macro_micro_F1(f1_result)
        # print('micro-F1: {0:.4f} - macro-F1: {1:.4f}'.format(micro_F1, macro_F1))
        if F > f:
            f = F
            tpreu = yu.argmax(axis=-1).reshape((1, size))[0]
            tprecue = ycue.argmax(axis=-1).reshape((1, size))[0]
            tcuevalue = ycue.max(axis=-1).reshape((1, size))[0]
    gold, pre, au_gold, au_pre, in_gold, in_pre = get_pre(tpreu, tprecue, tcuevalue, key)
    print('best F1:')
    all_result = scoreF(gold, pre, False)
    # f1_result = cal_F1_measure(gold, pre)
    # micro_F1, macro_F1 = cal_macro_micro_F1(f1_result)
    # print('micro-F1: {0:.4f} - macro-F1: {1:.4f}'.format(micro_F1, macro_F1))
    print('author:')
    au_result = scoreF(au_gold, au_pre, False)
    # f1_result = cal_F1_measure(au_gold,au_pre)
    # micro_F1, macro_F1 = cal_macro_micro_F1(f1_result)
    # print('micro-F1: {0:.4f} - macro-F1: {1:.4f}'.format(micro_F1, macro_F1))
    print('in source:')
    in_result = scoreF(in_gold, in_pre, False)
    # f1_result = cal_F1_measure(in_gold, in_pre)
    # micro_F1, macro_F1 = cal_macro_micro_F1(f1_result)
    # print('micro-F1: {0:.4f} - macro-F1: {1:.4f}'.format(micro_F1, macro_F1))

    return all_result, au_result, in_result


if __name__ == '__main__':
    resall = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    resau = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    resin = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    for i in range(10):
        all, author, inr = train_gan(i)
        for key in resall.keys():
            if key in all.keys():
                resall[key] += all[key]
            if key in author.keys():
                resau[key] += author[key]
            if key in inr.keys():
                resin[key] += inr[key]
    print('Uu:'+str(resall['Uu']/10)+'\tPR+:'+str(resall['PR+']/10)+'\tPS+'+str(resall['PS+']/10)+'\tCT-'
          +str(resall['CT-']/10)+ '\tPS-'+str(resall['PS-']/10)+ '\tPR-'+str(resall['PR-']/10)+ '\tCT+'
          +str(resall['CT+']/10)+'\tmicro'+str(resall['micro']/10)+'\tmacro'+str(resall['macro']/10))
    print('Uu:' + str(resau['Uu'] / 10) + '\tPR+:' + str(resau['PR+'] / 10) + '\tPS+' + str(resau['PS+'] / 10) + '\tCT-'
          + str(resau['CT-'] / 10) + '\tPS-' + str(resau['PS-'] / 10) + '\tPR-' + str(resau['PR-'] / 10) + '\tCT+'
          + str(resau['CT+'] / 10) + '\tmicro' + str(resau['micro'] / 10) + '\tmacro' + str(resau['macro'] / 10))
    print('Uu:' + str(resin['Uu'] / 10) + '\tPR+:' + str(resin['PR+'] / 10) + '\tPS+' + str(resin['PS+'] / 10) +
          '\tCT-'+ str(resin['CT-'] / 10) + '\tPS-' + str(resin['PS-'] / 10) + '\tPR-' + str(resin['PR-'] / 10)
          + '\tCT+'+ str(resin['CT+'] / 10) + '\tmicro' + str(resin['micro'] / 10) + '\tmacro' + str(resin['macro'] / 10))





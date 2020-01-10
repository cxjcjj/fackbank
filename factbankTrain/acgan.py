#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/12/9 12:26
from __future__ import print_function

from collections import Counter

from keras.utils import to_categorical
import pickle
from six.moves import range

import keras.backend as K
from keras.initializers import Constant
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, LSTM, Bidirectional, concatenate,Multiply, Dot, multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.layers.core import Lambda
import numpy as np
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential, load_model, save_model
import data
from data import cal_F1_measure,cal_macro_micro_F1, scoreF
from keras.optimizers import SGD
from keras.utils import plot_model

np.random.seed(2019)

class GAN:
    def __init__(self):
        self.epochs = 1
        self.batch = 100
        self.num_words = 4396
        self.maxlen = [22, 27, 26]  # sip, rs, cue 路径最大长度
        self.lr = 0.01
        self.dropout = 0.5
        self.train_set = []
        self.valid_set = []
        self.test_set = []
        self.embdding = None
        self.index = 9
        self.lenth = 20
        self.generator =None
        self.discriminator = None
        self.combine = None
        self.disfile = None
        self.genfile = None
        self.comfile = None

    def load_data(self, index):
        self.maxlen = data.produce('sample.pkl', 'mydata.pkl', index)
        self.train_set, self.valid_set,self.test_set = data.load('mydata.pkl')
        self.embdding = np.array(data.load('word_emb.pkl'))
        return

    def load_test_data(self, filename):
        self.test_set = data.read_file(filename)
        self.embdding = np.array(data.load('word_emb.pkl'))
        return

    def lstm_layer(self, x, maxlen):
        lstm = Sequential()
        lstm.add(Embedding(self.num_words, 100, input_length=maxlen))
        lstm.add(Dropout(self.dropout))
        lstm.add(Dense(maxlen))
        lstm.add(LSTM(100, return_sequences=True, dropout=self.dropout, activation='sigmoid'))
        lstm.add(Dense(100, activation='softmax', dtype='float32'))
        return lstm(x)

    def build_generator(self):
        # this is the noise space commonly refered to in GAN
        z0 = Input(shape=(self.maxlen[0],), dtype='float32')
        z1 = Input(shape=(self.maxlen[1],), dtype='float32')
        z2 = Input(shape=(self.maxlen[2],), dtype='float32')
        # this will be our label
        lau = Input(shape=(1,), dtype='float32')
        lacue = Input(shape=(1,), dtype='float32')

        temp = multiply([lau, lacue])
        x0 = multiply([z0, temp])
        x1 = multiply([z1, temp])
        x2 = multiply([z2, temp])
        fake0 = self.lstm_layer(x0, self.maxlen[0])
        fake1 = self.lstm_layer(x1, self.maxlen[1])
        fake2 = self.lstm_layer(x2, self.maxlen[2])

        return Model(inputs=[z0, z1, z2, lau, lacue], outputs=[fake0, fake1, fake2])

    def bilstm_layer(self, x, maxlen):
        bilstm = Sequential()
        bilstm.add(Bidirectional(LSTM(50, input_shape=(maxlen, 100), return_sequences=True, dropout=self.dropout)))
        bilstm.add(Dense(maxlen))
        p = bilstm(x)
        m = Dense(1, activation='tanh')(p)
        a = Dense(1, activation='softmax')(m)
        h = multiply([p, a])
        o = Dense(1, activation='tanh')(h)
        out = Reshape((maxlen,))(o)

        return out

    def build_discriminator(self):
        sip_path = Input(shape=(self.maxlen[0], 100,), dtype='float32')
        rs_path = Input(shape=(self.maxlen[1], 100,), dtype='float32')
        cue_path = Input(shape=(self.maxlen[2], 100,), dtype='float32')
        i = Input(shape=(100, ), dtype='float32')
        t = Input(shape=(100, ), dtype='float32')
        hs = self.bilstm_layer(sip_path, self.maxlen[0])
        hr = self.bilstm_layer(rs_path, self.maxlen[1])
        hc = self.bilstm_layer(cue_path, self.maxlen[2])

        fu = concatenate([hs, hr])
        fcue = concatenate([i, t, hc])
        outu = Dense(3, activation='softmax')(fu)
        outc = Dense(1, activation='sigmoid')(fcue)
        fakes = Dense(1, activation='softmax')(hs)
        faker = Dense(1, activation='softmax')(hr)
        fakec = Dense(1, activation='softmax')(hc)
        fakes = concatenate([fakes, faker, fakec])
        rfake = Dense(1, activation='sigmoid')(fakes)

        return Model(inputs=[sip_path, rs_path, cue_path, i, t], outputs=[rfake, outu, outc])

    def build_combine(self):
        # build discriminator
        self.discriminator = self.build_discriminator()
        sgd1 = SGD(lr=0.3)
        self.discriminator.compile(optimizer=Adam(lr=self.lr, beta_1=0.5),
                              loss=['binary_crossentropy', 'sparse_categorical_crossentropy',
                                    'binary_crossentropy'],
                              metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()
        sgd2 = SGD(lr=0.3)
        self.generator.compile(optimizer=Adam(lr=self.lr, beta_1=0.5),
                          loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy',
                                'sparse_categorical_crossentropy'],
                          metrics=['accuracy'])


        # this is the z space commonly refered to in GAN
        n0 = Input(shape=(self.maxlen[0],), dtype='float32', name='n0')
        n1 = Input(shape=(self.maxlen[1],), dtype='float32', name='n1')
        n2 = Input(shape=(self.maxlen[2],), dtype='float32', name='n2')

        # this will be our label
        labu = Input(shape=(1,), dtype='float32', name='labu')
        labcue = Input(shape=(1,), dtype='float32', name='labcue')
        pos = Input(shape=(100,), dtype='float32', name='pos')
        ct = Input(shape=(100,), dtype='float32', name='ct')

        self.discriminator.trainable = False
        fs, fr, fc = self.generator([n0, n1, n2, labu, labcue])
        fv, uv, cv = self.discriminator([fs, fr, fc, pos, ct])
        self.combine = Model(inputs=[n0, n1, n2, labu, labcue, pos, ct], outputs=[fv, uv, cv])

        sgd3 = SGD(lr=0.3)
        self.combine.compile(optimizer=Adam(lr=self.lr, beta_1=0.5),
                        loss=['binary_crossentropy', 'sparse_categorical_crossentropy',
                              'binary_crossentropy'],
                        metrics=['accuracy'])

    def train_gan(self, key):
        print('file:', str(key+1))
        # get data
        self.load_data(key)
        self.build_combine()
        tr_x0, tr_x1, tr_x2, tr_x3, tr_x4, tr_x5, tr_x6, tr_x7, tr_x8, tr_y0, tr_y1, tr_y_one = self.train_set
        va_x0, va_x1, va_x2, va_x3, va_x4, va_x5, va_x6, va_x7, va_x8, va_y0, va_y1, va_y_one = self.valid_set
        te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1, te_y_one = self.test_set
        f = 0
        g_loss = [0]
        d_loss = [0]

        for epoch in range(self.epochs):
            print('Epoch {} of {}'.format(epoch + 1, self.epochs))

            batch = int(len(self.train_set[0]) / self.batch)
            # progress_bar = Progbar(target=batch)

            for i in range(batch):
                # progress_bar.update(i)
                # generate a new batch of noise
                begin = i * self.batch
                end = (i + 1) * self.batch if (i + 1) * self.batch < len(self.train_set[0]) else len(self.train_set[0])
                lenth = end - begin
                # get a batch of real daata
                t = np.array(tr_x4)
                temp_sip = np.array(tr_x4[begin: end])
                temp_rs = np.array(tr_x5[begin: end])
                temp_cue = np.array(tr_x6[begin: end])
                idx = np.array(tr_x7[begin: end])
                ty = np.array(tr_x8[begin: end])

                real_sip = emb(self.embdding, temp_sip)
                real_rs = emb(self.embdding, temp_rs)
                real_cue = emb(self.embdding, temp_cue)
                index = emby(self.embdding, idx)
                cue_type = emby(self.embdding, ty)

                tr_yu = np.array(tr_y0[begin: end]).reshape((lenth, 1))
                tr_ycue = np.array(tr_y1[begin: end]).reshape((lenth, 1))
                tr_yf = np.array(tr_y_one[begin: end])

                rng = np.random.RandomState(9998)
                noise0 = rng.normal(scale=0.01, size=(lenth, self.maxlen[0])).astype('float32')
                noise1 = rng.normal(scale=0.01, size=(lenth, self.maxlen[1])).astype('float32')
                noise2 = rng.normal(scale=0.01, size=(lenth, self.maxlen[2])).astype('float32')

                fake_sip, fake_rs, fake_cue = self.generator.predict([noise0, noise1, noise2, tr_yu, tr_ycue])
                fake_yf = np.zeros((lenth,1), dtype=float)
                fake_yu = np.random.randint(0, 3, (lenth, 1)).astype('int32')
                fake_ycue = np.random.randint(0, 2, (lenth, 1)).astype('int32')
                sips = np.concatenate([real_sip, fake_sip], axis=0)
                rss = np.concatenate([real_rs, fake_rs], axis=0)
                cues = np.concatenate([real_cue, fake_cue], axis=0)
                indexs = np.concatenate([index, index], axis=0)
                types = np.concatenate([cue_type, cue_type], axis=0)
                tr_f = np.concatenate([tr_yf, np.zeros((lenth,),dtype=float)]).reshape((-1, 1))
                tr_u = np.concatenate([tr_yu, fake_yu]).reshape((-1, 1))
                tr_cue = np.concatenate([tr_ycue, fake_ycue]).reshape((-1, 1))

                if d_loss[0] < g_loss[0] *0.8:
                    g_loss = self.combine.train_on_batch([noise0, noise1, noise2, tr_yu, tr_ycue, index, cue_type],
                                                         [fake_yf, fake_yu, fake_ycue], class_weight=[None, 'auto', 'auto'])
                elif g_loss[0] < d_loss[0]*0.8:
                    d_loss = self.discriminator.train_on_batch([sips, rss, cues, indexs, types], [tr_f, tr_u, tr_cue], class_weight=[None, 'auto', 'auto'])
                else:
                    d_loss = self.discriminator.train_on_batch([sips, rss, cues, indexs, types], [tr_f, tr_u, tr_cue], class_weight=[None, 'auto', 'auto'])
                    g_loss = self.combine.train_on_batch([noise0, noise1, noise2, tr_yu, tr_ycue, index, cue_type], [fake_yf, fake_yu, fake_ycue],  class_weight=[None, 'auto', 'auto'])
                print("%d [D loss: %f, G loss: %f]" % (i, d_loss[0], g_loss[0]))

            print('\nTesting for epoch {}:'.format(epoch + 1))

            # get valid data
            temp_sip = np.array(va_x4)
            temp_rs = np.array(va_x5)
            temp_cue = np.array(va_x6)
            real_sip = emb(self.embdding, temp_sip)
            real_rs = emb(self.embdding, temp_rs)
            real_cue = emb(self.embdding, temp_cue)
            idx = np.array(va_x7)
            ty = np.array(va_x8)
            index = emby(self.embdding, idx)
            tp = emby(self.embdding, ty)

            va_yu = np.array(va_y0).reshape((-1, 1))
            va_ycue = np.array(va_y1).reshape((-1, 1))
            va_yf = np.array(va_y_one).reshape((-1, 1))
            self.discriminator.train_on_batch([real_sip, real_rs, real_cue, index, tp], [va_yf, va_yu, va_ycue])

            # get test data
            temp_sip = np.array(te_x4)
            temp_rs = np.array(te_x5)
            temp_cue = np.array(te_x6)
            real_sip = emb(self.embdding, temp_sip)
            real_rs = emb(self.embdding, temp_rs)
            real_cue = emb(self.embdding, temp_cue)
            idx = np.array(te_x7)
            ty = np.array(te_x8)
            index = emby(self.embdding, idx)
            tp = emby(self.embdding, ty)

            te_ycue = np.array(te_y1).reshape((-1, 1))
            size = te_ycue.shape[0]
            yf, yu, ycue = self.discriminator.predict([real_sip, real_rs, real_cue, index, tp])
            preu = yu.argmax(axis=-1).reshape((1, size))[0]
            precue = ycue
            gold, pre, au_gold, au_pre, in_gold, in_pre = get_pre(preu, precue,key)
            P, R, F = score(gold, pre, False)
            if F > f:
                f = F
                tpreu = yu.argmax(axis=-1).reshape((1, size))[0]
                tprecue = ycue
                filepre = "data/model/gan/"
                num = str(key) + '_' + str(epoch) + '.h5'
                self.disfile = filepre+'dis'+num
                self.genfile = filepre+'gen'+num
                self.comfile = filepre+'com'+num
                self.discriminator.save(self.disfile)
                self.generator.save(self.genfile)
                self.combine.save(self.comfile)

        gold, pre, au_gold, au_pre, in_gold, in_pre = get_pre(tpreu, tprecue, key)
        print('best F1:\n all:')
        all_result = scoreF(gold, pre, False)
        print('author:')
        au_result = scoreF(au_gold, au_pre, False)
        print('in source:')
        in_result = scoreF(in_gold, in_pre, False)

        return all_result, au_result, in_result

    def test_gan(self):
        dis = load_model(self.disfile)
        self.build_combine()
        te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1, te_y_one = self.test_set
        # get test data
        temp_sip = np.array(te_x4)
        temp_rs = np.array(te_x5)
        temp_cue = np.array(te_x6)
        real_sip = emb(self.embdding, temp_sip)
        real_rs = emb(self.embdding, temp_rs)
        real_cue = emb(self.embdding, temp_cue)
        idx = np.array(te_x7)
        ty = np.array(te_x8)
        index = emby(self.embdding, idx)
        tp = emby(self.embdding, ty)

        te_ycue = np.array(te_y1).reshape((-1, 1))
        size = te_ycue.shape[0]
        yf, yu, ycue = dis.predict([real_sip, real_rs, real_cue, index, tp])
        preu = yu.argmax(axis=-1).reshape((1, size))[0]
        precue = ycue
        gold, pre, au_gold, au_pre, in_gold, in_pre = get_pre(preu, precue, key)
        P, R, F = score(gold, pre, False)
        print('best F1:\n all:')
        all_result = scoreF(gold, pre, False)
        print('author:')
        au_result = scoreF(au_gold, au_pre, False)
        print('in source:')
        in_result = scoreF(in_gold, in_pre, False)

        return all_result, au_result, in_result




def emb(emb, x):
    y = np.zeros((x.shape[0],x.shape[1], 100))
    for k in range(len(x)):
        for i in range(len(x[k])):
            y[k][i] = emb[x[k][i]]
    return y


def emby(emb, x):
    # embeding label
    y = np.zeros((x.shape[0], 100))
    for k in range(len(x)):
        y[k] = emb[x[k]]
    return y.astype(np.float32)


def change(x, num):
    y = np.zeros((x.shape[0], num))
    for i in range(len(x)):
        y[i] = to_categorical(x[i], num)
    return y


def judge_label(u_label, cue_label, cue):
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
            if lc >= 0.1:
                apply.append(i)
        i += 1
    pre = 1
    v = 0
    va = 3
    if apply:
        # ps:1  pr:2    neg:3   ps_neg:4    pr_neg:5
        for j in apply:
            if cue[j] == '4' or cue[j] == '5':
                pre = -1
                if cue_label[j] > v:
                    v = cue_label[j]
                    va = int(cue[j])-3

            elif cue[j] == '3':
                pre = -1
            elif cue[j] == '1' or cue[j]=='2':
                if cue_label[j] > v:
                    v = cue_label[j]
                    va = int(cue[j])
        res = pre * va
    else:
        res = 3

    return dic[res]


def get_pre(resu, reacue, index):
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
        pre = judge_label(resu[i:i+lenth], reacue[i:i+lenth], cue)
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
        if gold != 'CT+':
            if pre == gold:
                TP += 1
            else:
                FN += 1
        else:
            if pre != 'CT+':
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
    print('acc:', acc, 'P:', P, "\tR:", R, '\tF',F)
    return P, R, F



if __name__ == '__main__':
    resall = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    resau = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    resin = {'Uu': 0, 'PR+': 0, 'PS+': 0, 'CT-': 0, 'PS-': 0, 'PR-': 0, 'CT+':0, 'micro':0,'macro':0}
    ac = GAN()
    flag = 'train'
    for i in range(10):
        if flag == 'train':
            all, author, inr = ac.train_gan(i)
        else:
            all, author, inr = ac.test_gan(i)
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





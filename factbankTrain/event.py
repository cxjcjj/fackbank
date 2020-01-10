#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2020/1/5 19:48

from deal import data
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Embedding, Dropout, LSTM, Bidirectional, \
    concatenate, Multiply, Dot, ZeroPadding1D, Conv1D, TimeDistributed
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import Sequential, Model, load_model
from keras.layers.core import Lambda
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import os
from collections import Counter
from keras.utils.np_utils import to_categorical

class LSTMCRF:
    def __init__(self):
        self.event_num = 0
        self.dim = 100
        self.label = {}
        self.label_size = 2
        self.feature_size = 3
        self.maxlen = 1
        self.train_len = 1
        self.sen_size = 200
        self.train_epochs = 50
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
        self.feature_flag = [1, 1, 1, 1, 1, 1]
        self.file = None

    def data(self):
        datas = data.Data()
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
        testf0 = []
        trainf0 = []
        validf0 = []
        gold = []
        for i in range(10):
            ins = self.file[i]
            if i == index:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    if len(words) >maxlen:
                        maxlen = len(words)
                    test_x.append(words)
                    test_y.append(labels)
                    testf0.append([t[1] for t in features])
                    gold.append(labels)

            elif i == k:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    if len(words) >maxlen:
                        maxlen = len(words)
                    valid_x.append(words)
                    valid_y.append(labels)
                    validf0.append([t[1] for t in features])
            else:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    if len(words) >maxlen:
                        maxlen = len(words)
                    train_x.append(words)
                    train_y.append(labels)
                    trainf0.append([t[1] for t in features])


        train_x = pad_sequences(train_x, maxlen)
        train_y = pad_sequences(train_y, maxlen)
        trainf0 = pad_sequences(trainf0, maxlen)
        test_x = pad_sequences(test_x, maxlen)
        test_y = pad_sequences(test_y, maxlen)
        testf0 = pad_sequences(testf0, maxlen)
        valid_x = pad_sequences(valid_x, maxlen)
        valid_y = pad_sequences(valid_y, maxlen)
        validf0 = pad_sequences(validf0, maxlen)

        self.gold = gold
        self.maxlen = maxlen
        self.test_x = test_x
        self.train_x = train_x
        self.valid_x = valid_x
        self.test_y = test_y
        self.train_y = train_y
        self.valid_y = valid_y
        self.test_f = testf0
        self.train_f = trainf0
        self.valid_f = validf0

    def build_model(self):
        word_input = Input(shape=(self.maxlen,), dtype='int32', name='word_input')
        word_emb = Embedding(input_dim=len(self.emb), output_dim=100, input_length=self.maxlen, weights=[self.emb],
                            trainable=False)(word_input)
        # bilstm
        bilstm = Bidirectional(LSTM(32, return_sequences=True))(word_emb)
        bilstm_d = Dropout(0.1)(bilstm)

        # cnn
        half_window_size = 2
        padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
        conv = Conv1D(nb_filter=50, filter_length=2 * half_window_size + 1, padding='valid')(padding_layer)
        conv_d = Dropout(0.1)(conv)
        dense_conv = TimeDistributed(Dense(50))(conv_d)

        # merge
        rnn_cnn_merge = concatenate([bilstm_d,dense_conv], axis=2)
        dense = TimeDistributed(Dense(self.label_size))(rnn_cnn_merge)

        outs = Dense(self.label_size, activation='softmax')(dense)
        # build model
        model = Model(input=[word_input], output=[outs])

        model.compile(optimizer='rmsprop',  # 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        self.model = model
        return model

    def train(self):
        tensorboard = TensorBoard(log_dir='data/log')
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = save_dir+'/event/best.hdf5'
        cw = {0: 1, 1: 3}
        checkpoint = ModelCheckpoint(filepath=filename, monitor='val_accuracy', mode='auto', save_best_only='True')

        callback_lists = [tensorboard, checkpoint]
        train_x = self.train_x
        train_f0 = self.train_f.reshape((-1, self.maxlen))

        valid_x = self.valid_x
        valid_f0 = self.valid_f.reshape((-1, self.maxlen))

        train_y = self.train_y.reshape((-1, self.maxlen, 1))  # np.eye(self.label_size)[self.train_y]
        valid_y = self.valid_y.reshape((-1, self.maxlen, 1))  # np.eye(self.label_size)[self.valid_y]
        self.model.fit(train_x, train_y,
                       batch_size=self.batch,
                       epochs=self.train_epochs,
                       validation_data=(valid_x, valid_y),
                       shuffle=1,
                       verbose=1,
                       class_weight='auto',
                       callbacks=callback_lists)

    def predict(self):
        test_x = self.test_x
        test_f0 = self.test_f.reshape((-1, self.maxlen))
        test_y = np.eye(self.label_size)[self.test_y]
        name = 'saved_models/event/'
        best_model_name = 'best.hdf5'
        self.best_model = load_model(os.path.join(name, best_model_name),
                                     custom_objects={'loss': 'sparse_categorical_crossentropy',
                                                     'accuracy': 'accuracy'})
        result = self.best_model.predict(test_x)
        self.result = result.argmax(axis=-1)
        self.score()

    def result_file(self, readfile, comfile, resultfile):
        outfile = open(comfile, 'w', encoding="utf-8")
        outresult = open(resultfile, 'w', encoding="utf-8")
        read = open(readfile, 'r', encoding='utf-8')
        lines = read.readlines()
        i = 0
        result = []
        gold = []
        for golds, pres in zip(self.gold, self.result):
            rpres = pres[len(pres) - len(golds):]
            for r, g in zip(rpres, golds):
                result.append(r)
                gold.append(g)
        for line in lines:
            if len(line) > 1:
                token = line.strip().split(self.split_word)
                word = token[0]
                label = token[-1]
                if self.label[gold[i]] != label:
                    print('error:', i)
                outresult.write(word + self.split_word + self.label[result[i]] + '\n')
                outfile.write(word + '\t' + label + '\t' + self.label[result[i]] + '\n')
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
        allgolds = [t for g in self.gold for t in g]
        label_counter = Counter(allgolds)
        allpres = [t for g in self.result for t in g]
        error = Counter(allpres)
        # Counter(计数器)是对字典的补充，用于追踪值的出现次数
        # most_common(a)截取指定位数的值，截取前a位的值
        for label, num in label_counter.most_common():
            print(label, '-----', num)
            # label是正确标签的数值
        print('pre:')
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
            for golds, pres in zip(self.gold, self.result):
                rpres = pres[len(pres) - len(golds):]
                for gold, pre in zip(golds, rpres):
                    if pre == key:
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
        for golds, pres in zip(self.gold, self.result):
            rpres = pres[len(pres) - len(golds):]
            for gold, pre in zip(golds, rpres):
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
        if TP + FN == 0:
            R = 0
        else:
            R = TP * 1.0 / (TP + FN)
        if P + R == 0:
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
    ls = LSTMCRF()
    ls.data()
    for i in range(10):
        # m = Max.MaxEnt()
        print("event" + str(i))
        ls.load_data(i)
        ls.build_model()
        ls.train()
        ls.predict()
        ls.result_file('data/read//event_' + str(i + 1) + '.txt',
            'data/result//compare_event_result_' + str(i + 1) + '.txt',
                            'data/result//event_result_' + str(i + 1) + '.txt')
        resp, resr, resf, p, r, f = ls.score()
        sum_p += p
        sum_r += r
        sum_f += f
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)







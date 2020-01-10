#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/12/13 17:35
from deal import data
from keras.layers import Flatten, Embedding, Dropout
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import os
from collections import Counter

class Den:
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
        test_f = []
        train_f = []
        valid_f = []
        self.i = index
        for i in range(10):
            ins = self.file[i]
            if i == index:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    for word, label, feature in zip(words, labels, features):
                        self.test_x.append(word)
                        self.test_y.append(label)
                        self.gold.append(label)
                        test_f.append(feature)

                        if len(feature[0]) > maxlen:
                            maxlen = len(feature[0])

            elif i == k:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    for word, label, feature in zip(words, labels, features):
                        self.valid_x.append(word)
                        self.valid_y.append(label)
                        valid_f.append(feature)
                        if len(feature[0]) > maxlen:
                            maxlen = len(feature[0])
            else:
                for sen in ins:
                    words = sen[0]
                    features = sen[1]
                    labels = sen[-1]
                    for word, label, feature in zip(words, labels, features):
                        self.train_x.append(word)
                        self.train_y.append(label)
                        train_f.append(feature)
                        if len(feature[0]) > maxlen:
                            maxlen = len(feature[0])
        feature_num = len(train_f[0])

        for k in range(len(train_f)):
            train_f[k][0].extend([0] * (maxlen - len(train_f[k][0])))
            temp = []
            for j in range(feature_num):
                if self.feature_flag[j]:
                    temp.extend(train_f[k][j])
            self.train_f.append(temp)
        for k in range(len(test_f)):
            test_f[k][0].extend([0] * (maxlen - len(test_f[k][0])))
            temp = []
            for j in range(feature_num):
                if self.feature_flag[j]:
                    temp.extend(test_f[k][j])
            self.test_f.append(temp)
        for k in range(len(valid_f)):
            valid_f[k][0].extend([0] * (maxlen - len(valid_f[k][0])))
            temp = []
            for j in range(feature_num):
                if self.feature_flag[j]:
                    temp.extend(valid_f[k][j])
            self.valid_f.append(temp)

        self.train_len = 1 + len(self.train_f[0])

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.emb), output_dim=100, input_length=self.train_len, weights=[self.emb],trainable=False))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.label_size, activation='softmax'))
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
        filename = save_dir+'/event/best'+str(self.i)+'.hdf5'
        cw = {0: 1, 1: 3}
        checkpoint = ModelCheckpoint(filepath=filename, monitor='val_accuracy', mode='auto', save_best_only='True')

        callback_lists = [tensorboard, checkpoint]

        train_x = np.concatenate((np.array(self.train_x).reshape((-1,1)), np.array(self.train_f)), axis=1)
        valid_x = np.concatenate((np.array(self.valid_x).reshape((-1,1)), np.array(self.valid_f)), axis=1)

        train_y = self.train_y
        valid_y = self.valid_y
        self.model.fit(train_x, train_y,
                       batch_size=self.batch,
                       epochs=self.train_epochs,
                       validation_data=(valid_x, valid_y),
                       shuffle=1,
                       verbose=0,
                       class_weight=cw,
                       callbacks=callback_lists)

    def predict(self):
        test_x = np.concatenate((np.array(self.test_x).reshape((-1, 1)), np.array(self.test_f)), axis=1)
        test_y = self.test_y
        best_model_name = 'saved_models/event/best'+str(self.i)+'.hdf5'
        print(best_model_name)
        self.best_model = load_model(best_model_name,
                                     custom_objects={'loss': 'sparse_categorical_crossentropy',
                                                     'accuracy': 'accuracy'})
        scores = self.best_model.evaluate(test_x, test_y)
        print('test loss:', scores[0])
        print('test accuracy:', scores[1])
        result = self.best_model.predict(test_x)
        self.result = result.argmax(axis=-1)

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
        if len(self.gold) != len(self.result):
            print('label list length Error')
            return None
        label_counter = Counter(self.gold)
        for label, num in label_counter.most_common():
            print(label, '-----', num)
        acc = accuracy_score(self.gold, self.result)
        f1 = f1_score(self.gold, self.result)
        r = recall_score(self.gold, self.result)
        p = precision_score(self.gold, self.result)
        print('acc', acc,'P:', p, "\tR:", r, '\tF', f1)
        print('acc:%.4f\tP:%.4f\tR:%.4f\tF:%.4f'%(acc, p, r,f1))
        return p, r, f1

if __name__ == "__main__":
    sum_p = 0
    sum_r = 0
    sum_f = 0
    ls = Den()
    ls.data()
    train = True
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
    print('sum\tP:%.4f\tR:%.4f\tF:%.4f'%(sum_p / 10,  sum_r / 10, sum_f / 10))







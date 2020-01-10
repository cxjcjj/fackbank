#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2020/1/8 20:11

#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:brick
# datetime:2019/11/15 14:49
import os
import pickle
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Input,  Activation, Reshape, Embedding, concatenate
from keras.models import  Model
from keras.layers.merge import  Dot
import numpy as np
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2
from deal import data

class CNNatt:
    # 初始化参数
    def __init__(self):
        self.event_num = 0
        self.dim = 100
        self.label_size = 2
        self.label = {}
        self.feature_size = 3
        self.max_word_size = 50  # 最大句子长度， 不足填充
        self.sen_size = 200
        self.train_epochs = 10
        self.batch = 100
        self.display_step = 100
        self.learn_rate = 0.01
        self.saver_dir = 'data/'
        self.train_x = []
        self.train_y = []
        self.train_f = []
        self.valid_x = []
        self.valid_y = []
        self.valid_f = []
        self.test_x = []
        self.test_y = []
        self.test_f = []
        self.result = []
        self.i = 0
        self.split_word = '&'
        self.split_feature = '*'
        self.model = None
        self.emb = None
        self.file =None

    def data(self):
        datas = data.Data()
        sip = datas.get_PSen('event_','')
        self.emb = datas.emb
        print(datas.sip_label_index.instance_index)
        temp_label = datas.sip_label_index.instance_index
        for key, value in temp_label.items():
            self.label[value] = key
        self.label_size = len(self.label)
        self.file = sip

    def load_data(self, index):
        """
        读取test set和train set
        :param index: 为test的的id
        :return:
        """
        self.i = index
        max = 0
        k = 9 if index != 9 else 8
        test_x = []
        train_x = []
        valid_x = []
        test_f = []
        train_f = []
        valid_f = []
        test_y = []
        train_y = []
        valid_y = []
        # file_emb 按照文件存储了10份数据， 第index份为test set，其余为train set

        for i in range(len(self.file)):
            for sen_emb in self.file[i]:
                word_f = sen_emb[0][:-1]
                label = sen_emb[0][-1]
                psen_f = sen_emb[1]
                if len(psen_f)>max:
                    max = len(psen_f)
                if i==index:
                    test_x.append(psen_f)
                    test_f.append(word_f)
                    test_y.append(label)
                elif i==k:
                    valid_x.append(psen_f)
                    valid_f.append(word_f)
                    valid_y.append(label)
                else:
                    train_x.append(psen_f)
                    train_f.append(word_f)
                    train_y.append(label)
        self.max_word_size = max
        # 不足最大长度， 进行扩充
        self.train_x = pad_sequences(train_x, max, padding='post')
        self.valid_x = pad_sequences(valid_x, max, padding='post')
        self.test_x = pad_sequences(test_x, max, padding='post')
        self.train_f = train_f
        self.train_y = train_y
        self.valid_f = valid_f
        self.valid_y = valid_y
        self.test_f = test_f
        self.test_y = test_y

        return


    def bulid_model(self):
        input_x = Input(shape=(self.max_word_size, ), dtype='float32')
        input_f = Input(shape=(self.feature_size,), dtype='float32')
        x= Embedding(input_dim=len(self.emb), output_dim=100, input_length=self.max_word_size, weights=[self.emb],
                             trainable=False)(input_x)
        f = Embedding(input_dim=len(self.emb), output_dim=100, input_length=self.feature_size, weights=[self.emb],
                            trainable=False)(input_f)
        y = Dense(self.dim, kernel_regularizer=l2(0.0003))(x)
        ym = Activation('tanh')(y)
        yt = Dense(self.max_word_size, kernel_regularizer=l2(0.0003))(ym)
        a = Activation('softmax')(yt)

        # def mult(x, y):
        #     return K.batch_dot(x, y, axes=1)

        ct = Dot(axes=1)([a,y])
        c = Dense(1, activation='tanh')(ct)
        cr = Reshape((self.max_word_size,))(c)
        lar = Reshape((300,))(f)
        con = concatenate([cr, lar], axis=1)
        out = Dense(2, activation='softmax',kernel_regularizer=l2(0.0003))(con)
        model = Model(inputs=[input_x, input_f], outputs=out)
        sgd = SGD(lr=0.3)
        model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        self.model = model
        return model

    def train(self):
        tensorboard = TensorBoard(log_dir='data/log')
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        filename = save_dir + '/sip/best' + str(self.i) + '.hdf5'

        checkpoint = ModelCheckpoint(filepath=filename, monitor='val_accuracy', mode='auto', save_best_only='True')

        callback_lists = [tensorboard, checkpoint]
        x = np.array(self.train_x)
        feature = np.array(self.train_f)
        y = np.array(self.train_y)
        valid_x = np.array(self.valid_x)
        valid_y = np.array(self.valid_y)
        valid_f = np.array(self.valid_f)

        self.model.fit([x, feature], y,
                       batch_size=self.batch,
                       epochs=self.train_epochs,
                       validation_data=([valid_x, valid_f], valid_y),
                       shuffle=1,
                       verbose=1,
                       class_weight = 'auto',
                       callbacks=callback_lists)

    def predict(self):
        test_x = np.array(self.test_x)
        test_f = np.array(self.test_f)
        test_y = np.array(self.test_y)

        name = 'saved_models/sip/'
        best_model_name = 'best' + str(self.i) + '.hdf5'
        self.best_model = load_model(os.path.join(name, best_model_name),
                                     custom_objects={'loss': 'sparse_categorical_crossentropy',
                                                     'accuracy': 'accuracy'})
        result = self.best_model.predict([test_x, test_f])
        pre = result.argmax(axis=-1)
        gold = test_y

        result = []
        for g,p in zip(pre, gold):
            result.append([g,p])
        self.result = result

        return

    def score(self, flag=True, result=[]):
        """
        计算PRF
        :param flag: 控制输出结果
        :param result: 结果存储 [gold, pre]
        :return: 返回PRF
        """
        if not result:
            result = self.result
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for gold, pre in result:
            if gold == 1:
                if pre == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pre == 1:
                    FP += 1
                else:
                    TN += 1
        acc = (TP + TN)*1.0 /(TP+TN+FP+FN)
        if TP+FP == 0:
            P = 0.0
        else:
            P = TP*1.0/(TP+FP)
        R = TP*1.0/(TP+FN)
        if P == 0.0 or R == 0.0:
            F = 0.0
        else:
            F = 2 * P*R/(P+R)
        if flag:
            print('acc:', acc)
            print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
            print('P:', P, "\tR:", R, '\tF', F)
        return P, R, F

    def result_file(self, file_feature):
        """
        输出对比文件、结果文件
        :param file_feature:{sen_id:{event_id:[[event, pos, up, label], [SPen]]}}
        :return:
        """
        outfile = open('data/result/sip_result_'+str(self.i + 1) + '.txt', 'w', encoding="utf-8")
        comfile = open('data/result/compare_sip_result_' + str(self.i + 1) + '.txt', 'w', encoding="utf-8")
        openfile = open('data/read/sip_'+str(self.i + 1) + '.txt', 'r', encoding="utf-8")

        feature = file_feature[self.i]  #{sen_id:{event_id:[[event, pos, up, label], [SPen]]}}
        sen_id = 0
        word_id = 0
        index = 0
        i = 0  # 文件行数

        for line in openfile.readlines():
            if len(line)>2:
                if sen_id in feature.keys():
                    temp_event = feature[sen_id]
                    if word_id in temp_event.keys():
                        outfile.write(line.strip().split(self.split_word)[0] +self.split_word+str(self.result[index][1])+'\n')
                        comfile.write(line.strip().split(self.split_word)[0] + '\t' + str(self.result[index][0])+ '\t' + str(self.result[index][1]) + '\n')
                        index += 1
                    else:
                        outfile.write(line.strip().split(self.split_word)[0]  + self.split_word + '0\n')
                        comfile.write(line.strip().split(self.split_word)[0] + '\t' + str(0)+ '\t' + str(0) + '\n')

                else:
                    outfile.write(line.strip() .split(self.split_word)[0] + self.split_word + '0\n')
                    comfile.write(line.strip().split(self.split_word)[0] + '\t' + str(0) + '\t' + str(0) + '\n')

                word_id += 1

            else:
                word_id = 0
                sen_id += 1
                outfile.write('\n')
                comfile.write('\n')
            i += 1


if __name__ == "__main__":
    # data = Data()
    # data.get_PSen('')
    sum_p = 0
    sum_r = 0
    sum_f = 0
    sum_tp = 0
    sum_fn = 0
    sum_fp = 0
    sum_tn = 0
    cnn = CNNatt()
    cnn.data()
    train = False
    for i in range(10):
        cnn.load_data(i)
        cnn.bulid_model()
        if train:
            cnn.train()
        cnn.predict()
        print('file'+str(i))
        p, r, f= cnn.score()
        with open('data/out/sip_feature' + '.pkl', 'rb') as file:
            file_feature = pickle.load(file)
        cnn.result_file(file_feature)
        sum_p += p
        sum_r += r
        sum_f += f
    print('sum\tP:', sum_p / 10, "\tR:", sum_r / 10, '\tF', sum_f / 10)
    print('sum\tP:%.4f\tR:%.4f\tF:%.4f' % (sum_p / 10, sum_r / 10, sum_f / 10))



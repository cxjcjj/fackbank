from __future__ import print_function
import pickle
from collections import Counter
import numpy as np

len1 = 22
len2 = 27
len3 = 26

def cal_F1_measure(correct_label_list, result_label_list):
    '''
    # 计算F1值
    返回结果字典, 格式 {标签:[TP,FP,FN,P,R,F1]}
    * correct_label_list: 正确的类别列表
    * result_label_list: 结果类别列表
    '''
    if len(correct_label_list) != len(result_label_list):
        print('label list length Error')
        return None
    # TP = 0
    # FP = 0
    # FN = 0
    result_dict = {}
    label_counter = Counter(correct_label_list)
    # Counter(计数器)是对字典的补充，用于追踪值的出现次数
    # most_common(a)截取指定位数的值，截取前a位的值
    for label, num in label_counter.most_common():
        print(label, '-----', num)
        # label是正确标签的数值
    for label in label_counter:
        print('class:', label)
        TP = FP = FN = 0
        FN_list = []
        FP_list = []
        for cor_label, res_label in zip(correct_label_list, result_label_list):
            # zip()函数用于将可以迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            if cor_label == label:
                if res_label == label:
                    TP += 1
                    # true positive
                else:
                    FN += 1
                    FN_list.append(res_label)
            else:
                if res_label == label:
                    FP += 1
                    FP_list.append(cor_label)
                    # false positive
        print('\tTP =', TP, '| FP =', FP, '| FN =', FN)
        print('\tFN =', Counter(FN_list))
        print('\tFP =', Counter(FP_list))
        result_list = []
        result_list.append(TP)
        result_list.append(FP)
        result_list.append(FN)
        if TP + FP == 0 or TP + FN == 0:
            print('can\'t calculate P and R')
            result_dict[label] = result_list
            continue
        P = TP / (TP + FP)  # precision P = (true positives)/ (true positive + false positive)
        R = TP / (TP + FN)  # recall
        result_list.append(P)
        result_list.append(R)
        if P + R == 0:
            print('can\'t calculate F1 measure')
            result_dict[label] = result_list
            continue
        F1 = (2 * P * R) / (P + R)
        result_list.append(F1)
        result_dict[label] = result_list
        print('\tprecision: {0:.4f} - recall: {1:.4f} - F1: {2:.4f}'.format(P, R, F1))
    return result_dict


def cal_macro_micro_F1(result_dict):
    '''
    # 计算Macro/Micro-F1值
    * result_dict: 结果字典, 格式 {标签:[TP,FP,FN,P,R]}
    '''
    macro_P = 0
    macro_R = 0
    TP_avg = 0
    FP_avg = 0
    FN_avg = 0
    for label, result_list in result_dict.items():
        if len(result_list) < 5:
            print(label, "cant't calculate this class:", label)
            continue
        if result_list[0] > 10:
            print("该类有效:", label)
            TP_avg += result_list[0]
            FP_avg += result_list[1]
            FN_avg += result_list[2]
            macro_P += result_list[3]
            macro_R += result_list[4]
    TP_avg /= 3
    FP_avg /= 3
    FN_avg /= 3
    micro_P = TP_avg / (TP_avg + FP_avg)
    micro_R = TP_avg / (TP_avg + FN_avg)
    micro_F1 = (2 * micro_P * micro_R) / (micro_P + micro_R)
    macro_P /= 3
    macro_R /= 3
    macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
    # macro_F1 = macro_F1 / 3
    print('micro-F1: {0:.4f} - macro-F1: {1:.4f}'.format(micro_F1, macro_F1))
    return micro_F1, macro_F1

def scoreF(golds, pres, flag=True):
    if len(golds) != len(pres):
        print('label list length Error')
        return None
    label_counter = Counter(golds)
    # Counter(计数器)是对字典的补充，用于追踪值的出现次数
    # most_common(a)截取指定位数的值，截取前a位的值
    for label, num in label_counter.most_common():
        print(label, '-----', num)
        # label是正确标签的数值

    for label in label_counter:
        print('class:', label)
    labels = ['Uu', 'CT+', 'CT-', 'PR+', 'PS+']
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    n = len(labels)
    TP_C = [0] * n
    FP_C = [0] * n
    TN_C = [0] * n
    FN_C = [0] * n
    P_C = [0] * n
    R_C = [0] * n
    F_C =[0] * n

    for i in range(n):
        for gold, pre in zip(golds, pres):
            if pre == labels[i]:
                pre = 1
            else:
                pre = 0
            if gold == labels[i]:
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

    macroP = 0
    macroR = 0
    macroF = 0
    iTP = 0
    iFN = 0
    iFP = 0
    result ={}
    for i in range(n):
        P_C[i] = TP_C[i]*1.0/(TP_C[i]+FP_C[i]) if TP_C[i]+FP_C[i]>0 else 0
        R_C[i] = TP_C[i]*1.0/(TP_C[i]+FN_C[i]) if TP_C[i]+FN_C[i]>0 else 0
        F_C[i] = 2 * P_C[i]*R_C[i]/(P_C[i]+R_C[i]) if P_C[i]+R_C[i]>0 else 0
        print('class'+str(labels[i])+'\tTP:', str(TP_C[i]), '\tTN:', str(TN_C[i]), '\tFP', str(FP_C[i]), '\tFN', str(FN_C[i]) )
        print('\t'+'P:', P_C[i], "\tR:", R_C[i], '\tF', F_C[i])
        result[labels[i]] = F_C[i]
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
        result['macro'] = macroF
        result['micro'] = microR

    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    P = TP*1.0/(TP+FP)
    R = TP*1.0/(TP+FN)
    F = 2 * P*R/(P+R)
    if flag:
        print('acc:', acc)
        print('TP:', TP, '\tTN:', TN, '\tFP', FP,'\tFN', FN)
    print('P:', P, "\tR:", R, '\tF',F)
    return result

def read_file(filename):
    with open(filename, 'rb') as file:
        file_feature = pickle.load(file)
    feature_sise = 9
    label_size = 3
    te = []
    for k in range(feature_sise + label_size):
        te.append([])
    for feature, label in file_feature:
        for j in range(len(feature)):
            te[j].append(list(map(int,feature[j])))
        te[-3].append(label[0])
        te[-2].append(label[1])
        te[-1].append(1)

    te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1, te_y_one = te
    te_x4 = pad(te_x4, len1)
    te_x5 = pad(te_x5, len2)
    te_x6 = pad(te_x6, len3)

    data = (te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1, te_y_one)
    return data

def read(filename, index):
    with open(filename, 'rb') as file:
        file_feature = pickle.load(file)
    feature_sise = 9
    label_size = 3
    tr = []
    va = []
    te = []
    for k in range(feature_sise+label_size):
        tr.append([])
        va.append([])
        te.append([])

    v_id = 9 if index!= 9 else 8
    maxlen1 = 0
    maxlen2 = 0
    maxlen3 = 0
    sample_train = []
    for i in range(10):
        if i == index:
            sample_test = file_feature[i]
        elif i == v_id:
            sample_va = file_feature[i]
        else:
            sample_train += file_feature[i]
        np.random.shuffle(sample_train) #打乱 用于训练
    for feature, label in sample_test:
        if len(feature[4]) > maxlen1:
            maxlen1 = len(feature[4])
        if len(feature[5]) > maxlen2:
            maxlen2 = len(feature[5])
        if len(feature[6]) > maxlen3:
            maxlen3 = len(feature[6])
        for j in range(len(feature)):
            te[j].append(list(map(int,feature[j])))
        te[-3].append(label[0])
        te[-2].append(label[1])
        te[-1].append(1)
    for feature, label in sample_va:
        if len(feature[4]) > maxlen1:
            maxlen1 = len(feature[4])
        if len(feature[5]) > maxlen2:
            maxlen2 = len(feature[5])
        if len(feature[6]) > maxlen3:
            maxlen3 = len(feature[6])
        for j in range(len(feature)):
            va[j].append(list(map(int,feature[j])))
        va[-3].append(label[0])
        va[-2].append(label[1])
        va[-1].append(1)
    for feature, label in sample_train:
        if len(feature[4]) > maxlen1:
            maxlen1 = len(feature[4])
        if len(feature[5]) > maxlen2:
            maxlen2 = len(feature[5])
        if len(feature[6]) > maxlen3:
            maxlen3 = len(feature[6])
        for j in range(len(feature)):
            tr[j].append(list(map(int,feature[j])))
        tr[-3].append(label[0])
        tr[-2].append(label[1])
        tr[-1].append(1)
    return tr, va, te, maxlen1, maxlen2, maxlen3

def pad(x, lenth):
    for i in range(len(x)):
        if len(x[i])<lenth:
            for j in range(lenth - len(x[i])):
                x[i].append(0)

    return x

def produce(readfile, savefile, index):
    tr, va, te, maxlen1, maxlen2, maxlen3 = read(readfile, index)
    tr_x0, tr_x1, tr_x2, tr_x3, tr_x4, tr_x5, tr_x6, tr_x7, tr_x8, tr_y0, tr_y1 , tr_y_one= tr
    va_x0, va_x1, va_x2, va_x3, va_x4, va_x5, va_x6, va_x7, va_x8, va_y0, va_y1 , va_y_one= va
    te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1 , te_y_one= te
    tr_x4 = pad(tr_x4, maxlen1)
    tr_x5 = pad(tr_x5, maxlen2)
    tr_x6 = pad(tr_x6, maxlen3)
    va_x4 = pad(va_x4, maxlen1)
    va_x5 = pad(va_x5, maxlen2)
    va_x6 = pad(va_x6, maxlen3)
    te_x4 = pad(te_x4, maxlen1)
    te_x5 = pad(te_x5, maxlen2)
    te_x6 = pad(te_x6, maxlen3)

    data = ((tr_x0, tr_x1, tr_x2, tr_x3, tr_x4, tr_x5, tr_x6, tr_x7, tr_x8, tr_y0, tr_y1 , tr_y_one),
            (va_x0, va_x1, va_x2, va_x3, va_x4, va_x5, va_x6, va_x7, va_x8, va_y0, va_y1 , va_y_one),
            (te_x0, te_x1, te_x2, te_x3, te_x4, te_x5, te_x6, te_x7, te_x8, te_y0, te_y1 , te_y_one))
    f = open(savefile, 'wb')
    pickle.dump(data, f)
    f.close()
    return [maxlen1, maxlen2, maxlen3]


def load(path):
    f = open(path, 'rb')
    files = pickle.load(f)
    f.close()
    return files


if __name__ == '__main__':
    produce('sample.pkl', 'mydata.pkl', 0)
    p = [1,2,3,4,0,2,3,4]
    g = [2,3,3,0,0,2,4,4]
    scoreF(g,p)
    cal_F1_measure(g,p)

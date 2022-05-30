# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os
import json
'''
将选出的最长序列做标准化处理并存在pred_data文件夹下用于模型预测验证
前五项是id,后面七个是特征值,不包含资源请求值
'''

#train_pth = r'/home/wxh/lstmDnn/anomaly/cache/array64_train.dat.npy'
train_pth = r'/home/wxh/lstmDnn/anomaly/cache/array64_train_6feature.dat.npy'
test_list = glob.glob(r'/home/wxh/google_trace/cache/testData/*/*.csv')
train_data = np.load(train_pth)
c = np.unique(train_data.reshape(-1,train_data.shape[-1]), axis = 0)[:, 5:-3]
mean, std = np.mean(c, axis = 0), np.std(c, axis=0)
jsonfile = r'/home/wxh/lstmDnn/anomaly/google_trace_config.json'
dataspecs = json.load(open(jsonfile, 'r'))
feature = dataspecs['feature']
for test_pth in test_list:
    #test_pth = r'/home/wxh/google_trace/cache/testData/3769734721/0.csv'
    #pred_pth_data = r'/home/wxh/lstmDnn/anomaly/pred_data/3769734721_0.csv'
    pred_pth_data = r'/home/wxh/lstmDnn/anomaly/pred_data/test/' + '_'.join(test_pth.split('/')[-2:])
    #test_data = pd.read_csv(test_pth, header=None).iloc[:,[0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 15, 16]]
    test_data = pd.read_csv(test_pth, header=None).iloc[:, feature]
    test_to_norm = np.array(test_data)[:, 5:-3]
    test_data_normal = (test_to_norm - mean)/std
    test_data.iloc[:, 5:-3] = test_data_normal
    test_data.to_csv(pred_pth_data, header=None, index=None)
    print('saved ' + pred_pth_data)


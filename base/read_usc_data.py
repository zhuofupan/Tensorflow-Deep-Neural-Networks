# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:17:57 2018

@author: Administrator
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

csv_dir = '../dataset/Urban Sound Classification'
sub_dir = '../saver/USC'

def preprocess(train_X=None,test_X=None,preprocessing='st'):
    if preprocessing=='st': # Standardization（标准化）
        prep = StandardScaler() # 标准化
    if preprocessing=='mm': # MinMaxScaler (归一化)
        prep = MinMaxScaler() # 归一化
    train_X = prep.fit_transform(train_X)
    test_X = prep.transform(test_X)
    return train_X,test_X


def read_data(meth='stat'): # stat | spec
    
    str_='('+meth+').csv'
    
    train_X = np.loadtxt(os.path.join(os.path.abspath(csv_dir),'Train_X'+str_),  # load 训练集样本
                         dtype = np.float32, delimiter=',')
    train_Y = np.loadtxt(os.path.join(os.path.abspath(csv_dir),'Train_Y'+str_),  # load 训练集标签
                         dtype = np.int32, delimiter=',')
    test_X = np.loadtxt(os.path.join(os.path.abspath(csv_dir),'Test_X'+str_),    # load 测试集样本
                        dtype = np.float32, delimiter=',')
    
    train_X, test_X = preprocess(train_X, test_X)
    
    return train_X, train_Y, test_X


def submission(predict):
    if(len(predict.shape)==2):
        label = np.argmax(predict,axis=1)
    else:
        label = predict
    label_str = list()
    test = pd.read_csv(os.path.join(os.path.abspath(csv_dir), 'test.csv'))               # load 测试集标签序号ID
    encoder = np.loadtxt(os.path.join(os.path.abspath(csv_dir), 'label_encoder.csv'),    # load 标签转换格式
                         dtype=np.str,delimiter=',')
    for i,lb in enumerate(label):
          label_str.append(encoder[lb])
    test['Class'] = label_str
    if not os.path.exists(os.path.join(os.path.abspath(sub_dir))): os.makedirs(os.path.join(os.path.abspath(sub_dir)))
    test.to_csv(os.path.join(os.path.abspath(sub_dir),'sub01[usc].csv'), index=False)         # save 测试集标签
    return test
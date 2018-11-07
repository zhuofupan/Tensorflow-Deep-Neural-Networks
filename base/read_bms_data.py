# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:20:39 2018

@author: Administrator
"""
import os
import sys
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
import tensorflow as tf
from dbn import DBN

###########################
#        参数设定         #
##########################

csv_dir = '../dataset/Big Mart Sales III'
sub_dir = '../saver/BMS'
ps = 'st'

###########################
#       预处理函数        #
##########################

def preprocess(train_X=None,test_X=None,preprocessing='st'):
    if preprocessing=='st': # Standardization（标准化）
        prep = StandardScaler() # 标准化
    if preprocessing=='mm': # MinMaxScaler (归一化)
        prep = MinMaxScaler() # 归一化
    train_X = prep.fit_transform(train_X)
    if test_X is not None:
        test_X = prep.transform(test_X)
    return train_X,test_X,prep

###########################
#     读取数据集<csv>     #
##########################

def word2int(X):
    X = np.array(X.tolist())
    lb = LabelEncoder()
    X = keras.utils.np_utils.to_categorical(lb.fit_transform(X))
    X = np.array(np.argmax(X,axis = 1).reshape(-1,1),dtype=np.float32)
    return X

def col_vaule(df,i):
    return np.array(df.iloc[:,i].values.reshape(-1,1))

# 读取不完备数据集
def make_datasets():
    
    # 读取数据集
    def make_data(tp):
        
        print(tp + " DATA...")
        df = pd.read_csv(os.path.join(os.path.abspath(csv_dir), tp+'.csv')) # load 训练集/测试集标签序号ID
        
        value_list = [3,5]
        word_list = [2,4,6,7,9,10]
        
        data = list()
        for i in value_list:
            data.append(col_vaule(df,i))
        for i in word_list:
            data.append(word2int(df.iloc[:,i]))
        col_8 = word2int(df.iloc[:,8])
        for i in range(len(col_8)):
            if col_8[i] == 3:
                col_8[i] = np.nan
        
        data = np.concatenate(data,axis = 1)
        data = np.concatenate((data,col_vaule(df,1),col_8),axis = 1)
        print(" --> finished!")
        
        # 第1,8列需要数据补全
        if tp == 'Train':
            y = np.array(df.iloc[:,11].values).reshape(-1,1)
            return data, y
        else:
            return data
    
    tp = 'Train'
    train_X, train_Y = make_data(tp)
    tp = 'Test'
    test_X = make_data(tp)
    
    return train_X, train_Y, test_X



###########################
#     处理不完备数据集     #
###########################

def split_datasets(train_X, test_X):   
    
    train_f8, test_f8, _ = preprocess(train_X[:,:8],test_X[:,:8],preprocessing=ps)      # 前8维变量标准化<预处理>
    train_X = np.concatenate((train_f8,train_X[:,8:]),axis = 1)
    test_X = np.concatenate((test_f8,test_X[:,8:]),axis = 1)
    
    def split_data(data):    
        """
            将数据集分割为 “完备集” 与 “不完备集”
            data89:
                -> 8,9
                    -> un,com_x,com_y
        """
        data89 = list()
        
        col = [8,9]
        for t in range(2):
            uncom_x = list()
            com_x = list()
            com_y = list()
            j = col[t]
            for i in range(len(data)):
                if np.isnan(data[i,j]):
                    uncom_x.append(data[i,:8].reshape(1,-1))
                else:
                    com_x.append(data[i,:8].reshape(1,-1))
                    com_y.append(data[i,j])
                    
            uncom_x = np.concatenate(uncom_x,axis = 0)
            com_x = np.concatenate(com_x,axis = 0)
            com_y = np.array(com_y)
            if j == 9:
                # one_hot:
                lb = LabelEncoder()
                com_y = keras.utils.np_utils.to_categorical(lb.fit_transform(com_y))
            else:
                com_y = com_y.reshape(-1,1)
                    
            data89.append([com_x,com_y,uncom_x])
        
        return data89
            
    train_dataset = split_data(train_X)
    test_dataset = split_data(test_X)
    
    datasets = list()
    for t in range(2):
        train_com_x  = np.array(train_dataset[t][0],dtype = np.float32)
        train_com_y  = np.array(train_dataset[t][1],dtype = np.float32)
        train_uncom_x = np.array(train_dataset[t][2],dtype = np.float32)
        
        test_com_x  = np.array(test_dataset[t][0],dtype = np.float32)
        test_com_y  = np.array(test_dataset[t][1],dtype = np.float32)
        test_uncom_x = np.array(test_dataset[t][2],dtype = np.float32)
        
        data = [train_com_x,test_com_x,train_com_y,test_com_y,train_uncom_x,test_uncom_x]
        datasets.append(data)
    return datasets

###########################
#         数据补全         #
###########################

def completion_by_prediction(t):
    train_com_x,test_com_x,train_com_y,test_com_y,train_uncom_x,test_uncom_x = datasets[t]
    train_X = np.concatenate((train_com_x,test_com_x),axis = 0)
    train_Y = np.concatenate((train_com_y,test_com_y),axis = 0)
    if t == 0: train_Y, _ ,prep = preprocess(train_Y,None,'mm')
    x_dim = train_X.shape[1]
    y_dim = train_Y.shape[1]
    if t == 0: 
        use_for='prediction'
        hidden_act_func = ['tanh','gauss']
        output_act_func = 'affine'
        lr = 1e-3
    else: 
        use_for='classification'
        hidden_act_func = ['tanh','gauss']
        output_act_func = 'softmax'
        lr = 1e-3
    tf.reset_default_graph()
    classifier = DBN(
                 hidden_act_func=hidden_act_func,
                 output_act_func=output_act_func,
                 loss_func='mse', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[x_dim, x_dim*40, x_dim*20, x_dim*10, x_dim*2, y_dim],
                 lr=lr,
                 use_for=use_for,
                 bp_algorithm='rmsp',
                 epochs=240,
                 batch_size=32,
                 dropout=0.08,
                 units_type=['gauss','gauss'],
                 rbm_lr=1e-4,
                 rbm_epochs=60,
                 cd_k=1,
                 pre_train=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    classifier.train_model(train_X = train_X, 
                           train_Y = train_Y, 
                           sess = sess)
    train_uncom_y = sess.run(classifier.pred,feed_dict={
            classifier.input_data: train_uncom_x,
            classifier.keep_prob: 1.0})
    test_uncom_y = sess.run(classifier.pred,feed_dict={
            classifier.input_data: test_uncom_x,
            classifier.keep_prob: 1.0})
    
    if t == 0:
        train_uncom_y = prep.inverse_transform(train_uncom_y.reshape(-1,1))
        test_uncom_y = prep.inverse_transform(test_uncom_y.reshape(-1,1))
    elif t == 1:
        train_uncom_y = np.argmax(train_uncom_y,axis = 1).reshape(-1,1)
        test_uncom_y = np.argmax(test_uncom_y,axis = 1).reshape(-1,1)
    return [train_uncom_y, test_uncom_y]

def recon_X():
    
    def recon_(data0,data8,data9):
        p8 = 0
        p9 = 0
        for i in range(len(data0)):
            if np.isnan(data0[i,8]):
                data0[i,8] = data8[p8]
                p8 = p8 + 1
            if np.isnan(data0[i,9]):
                data0[i,9] = data9[p9]
                p9 = p9 + 1
        return data0
    
    train_x = recon_(train_X,train_x8,train_x9)
    test_x = recon_(test_X,test_x8,test_x9)
    
    return train_x, test_x


###########################
#        读取与提交        #
###########################
    

def read_data():
    
    train = np.loadtxt(os.path.join(os.path.abspath(csv_dir),'train[cplt].csv'),  # load 训练集样本
                         dtype = np.float32, delimiter=',')
    test = np.loadtxt(os.path.join(os.path.abspath(csv_dir),'test[cplt].csv'),  # load 训练集标签
                         dtype = np.int32, delimiter=',')
    
    train_X = np.array(train[:,:-1],dtype = np.float32)
    train_Y = np.array(train[:,-1].reshape(-1,1),dtype = np.float32)
    test_X = np.array(test,dtype = np.float32)
    
    train_X, test_X, _ = preprocess(train_X, test_X)
    
    return [train_X, train_Y, test_X]


def submission(predict):
    df = pd.read_csv(os.path.join(os.path.abspath(csv_dir), 'Submission.csv'))
    df['Item_Outlet_Sales'] = predict
    if not os.path.exists(os.path.join(os.path.abspath(sub_dir))): os.makedirs(os.path.join(os.path.abspath(sub_dir)))
    df.to_csv(os.path.join(os.path.abspath(sub_dir),'sub01[bms].csv'), index=False)         # save 测试集标签

if __name__ == "__main__":
    
    # 读取不完备数据集
    train_X, train_Y, test_X = make_datasets()
    print(np.isnan(train_X[3,9]))
    print(np.isnan(train_X[7,8]))
    
    # 分割完备/不完备数据集
    datasets = split_datasets(train_X, test_X)
    
    # 用完备数据集预测不完备数据集
    train_x8, test_x8 = completion_by_prediction(0)
    train_x9, test_x9 = completion_by_prediction(1)
    
    # 合并数据集
    train_X,test_X = recon_X()
    
    # 保存数据集
    np.savetxt(os.path.join(os.path.abspath(csv_dir),'train[cplt].csv'),np.concatenate((train_X,train_Y),axis = 1),
               fmt='%.6f',delimiter=",")
    np.savetxt(os.path.join(os.path.abspath(csv_dir),'test[cplt].csv'),test_X,
               fmt='%.6f',delimiter=",")

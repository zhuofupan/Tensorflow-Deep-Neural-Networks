# -*- coding: utf-8 -*-
"""Functions for downloading and reading TE data."""
import numpy as np
import random

def extract_data_x(f):
    print('Extracting', f)
    data=[]
    row=0
    with open(f,'r',encoding="utf-8") as file:
        for line in file:
            l=line.encode('utf-8').decode('utf-8-sig').split(',')    #按,划分
            data+=l
            row=row+1
    column=int(len(data)/row)
    data=np.asarray(data,dtype='float32')
    data=data.reshape(row,column)
    return data

def shuffle_data(data_x,data_y):
    dic=list(zip(data_x,data_y))  #zip形成字典      
    random.shuffle(dic) #这个是将字典进行随机化排列
    data_x,data_y=map(list,zip(*dic))#将dic拆分成两个list
    data_x=np.asarray(data_x,dtype='float32')
    data_y=np.asarray(data_y,dtype='float32')
    return data_x,data_y

def make_data_sets(data_dir,
                   data_type,
                   one_hot=False):
    if data_type == 'train':
        data_set_x=[]
        data_set_y=[]
    else:
        data_set_x=list()
        data_set_y=list()
    for i in range(22):
        # 读文件
        if data_type == 'train':
            X = data_dir + '/d' + str(i) + '.csv'
        else:
            X = data_dir + '/d' + str(i) + '_te.csv'
        # 提取X
        data_x = extract_data_x(X)
        n = data_x.shape[0]
        # 生成Y
        if one_hot:
            data_y = np.fromfunction(lambda j,k:(i==k).astype(int),(n,22))
        else:
            data_y = np.full(n,i).reshape(n,1)
        # 合并各类数据集
        if data_type == 'train':
            if i>0:
                data_set_x = np.concatenate((data_set_x,data_x),axis=0)
                data_set_y = np.concatenate((data_set_y,data_y),axis=0)
            else:
                data_set_x = data_x
                data_set_y = data_y
        # 加入列表
        else:
            data_set_x.append(data_x)
            data_set_y.append(data_y)
    return data_set_x,data_set_y

def read_data_sets(data_dir,
                   one_hot=False,
                   shuffle=False):
    train_x,train_y = make_data_sets(data_dir,'train',one_hot)
    test_x,test_y = make_data_sets(data_dir,'test',one_hot)
  
    if shuffle:
        train_x,train_y = shuffle_data(train_x,train_y)
        #test_x,test_y = shuffle_data(test_x,test_y)
    
    return train_x,train_y,test_x,test_y

if __name__ == "__main__":
    train_x,train_y,test_x,test_y=read_data_sets('../../TE_csv',one_hot=True,shuffle=True)
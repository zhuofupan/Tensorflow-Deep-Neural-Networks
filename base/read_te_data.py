# -*- coding: utf-8 -*-
"""Functions for downloading and reading TE data."""
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest, chi2,f_classif,mutual_info_classif

#######################
#    make_data_set    #
#######################
    
def shuffle_data(data_x,data_y):
    """
    功能：对数据集进行洗牌
    """
    dic=list(zip(data_x,data_y))  #zip形成字典      
    random.shuffle(dic) #这个是将字典进行随机化排列
    data_x,data_y=map(list,zip(*dic))#将dic拆分成两个list
    data_x=np.asarray(data_x,dtype='float32')
    data_y=np.asarray(data_y,dtype='float32')
    
    return data_x,data_y


def get_origin_sets(data_dir,
                    data_type='train'):
    """
    功能：读取文件，返回原始数据集
    返回：train：返回合并各文件数据得到的X矩阵和Y向量
         test：返回X和Y的list
    """
    data_set_x=[]
    data_set_y=[]
    data_list_x=list()
    data_list_y=list()
    for i in range(22):
        # 读文件
        if data_type == 'train':
            file = data_dir + '/d' + str(i) + '.csv'
        else:
            file = data_dir + '/d' + str(i) + '_te.csv'
        # 提取X（从文件）
        print('Extracting', file)
        data_x=[]
        with open(file,'r',encoding="utf-8") as f:
            for j,line in enumerate(f):
                l=np.asarray(line.encode('utf-8').decode('utf-8-sig').split(','),dtype='float32').reshape(1,-1)    #按,划分
                if j==0: data_x=l
                else: data_x = np.concatenate((data_x,l),axis=0)
        # 生成Y（非one-hot）
        n = data_x.shape[0]
        data_y = np.full(n,i).reshape(n,1)
        # 合并各类数据集
        if i>0:
            data_set_x = np.concatenate((data_set_x,data_x),axis=0)
            data_set_y = np.concatenate((data_set_y,data_y),axis=0)
        else:
            data_set_x = data_x
            data_set_y = data_y
        # 加入列表
        data_list_x.append(data_x)
        data_list_y.append(data_y)
        
    if data_type == 'train': return data_set_x,data_set_y
    else: return data_list_x,data_list_y
    
    
def get_dynamic_datas(in_type,
                      out_type,
                      deal_type,
                      X,
                      select_dim,
                      dynamic,
                      one_hot):
    """
    参数：dim=52，fault=22，n_train=480，n_test=800
    返回：train：得到特征选择后的动态数据集
         train_x[(480-dynamic+1)*22 , dynamic*52] , train_y[(480-dynamic+1)*22 , 22(1)]
         
         test：得到动态数据集列表
         test_x[22 , 800-dynamic+1 , dynamic*52] , train_y[22, 800-dynamic+1 , 22(1)]
         
         delete：得到特征删减后的数据
         data_x[fault , n_sampels-dynamic+1 , dynamic*k_best] , data_y[fault, n_sampels-dynamic+1 , fault(1)]
    """
    if out_type=='array':
        data_x=[]
        data_y=[]
    else: 
        data_x=list()
        data_y=list()
    for i in range(22):
        # 原始数据
        if in_type=='array': file=X[i*n_train:(i+1)*n_train]
        else: file=X[i]
        # 特征选择：
        if deal_type!='no' and len(select_dim)>0:
            se_dim_list = select_dim[i].tolist()
            no_se_dim_list=list()
            for j in range(52):
                if j not in se_dim_list:
                    if deal_type=='0': file[:,j]=np.random.rand(file.shape[0]) # 过滤某些特征，让其值为噪声（bset_k>0）
                    if deal_type=='noise': file[:,j]=0. # 过滤某些特征，让其值为0（bset_k>0）
                    if deal_type=='delete': no_se_dim_list.append(j)
            if deal_type=='delete': file=np.delete(file,no_se_dim_list,axis=1) # 删除未选择的特征（OCON=True）
        # 动态数据集X
        dy_x=[]
        for r in range(file.shape[0]):
            if r+1>=dynamic:
                start=r+1-dynamic
                end=r+1
                nl=file[start:end].reshape(1,-1)
                if r+1==dynamic: dy_x=nl
                else: dy_x = np.concatenate((dy_x,nl),axis=0)
        # 动态数据集Y
        n=dy_x.shape[0]
        if one_hot:
            dy_y = np.fromfunction(lambda j,k:(k==i).astype(int),(n,22)) #.astype(int)
        else:
            dy_y = np.full(n,i).reshape(n,1)
        # data_x
        if out_type=='array':
            if i==0: 
                data_x = dy_x
                data_y = dy_y
            else: 
                data_x = np.concatenate((data_x,dy_x),axis=0)
                data_y = np.concatenate((data_y,dy_y),axis=0)
        else: 
            data_x.append(dy_x)
            data_y.append(dy_y)
            
    return data_x,data_y


def gene_net_datas(data_dir,
                   preprocessing='mm',
                   one_hot=True,
                   shuffle=True,
                   # 考虑动态数据集
                   dynamic=40,
                   # 用于特征选择
                   select_method='chi2',
                   k_best=0,
                   # 一类一网络
                   ocon=False):
    """
    返回：train_x,train_y 矩阵（OCON=Flase）/列表（OCON=True）
         test_lx,test_ly 列表
    """
    # 得到原始数据集
    train_x,train_y = get_origin_sets(data_dir,data_type='train')
    test_lx,test_ly = get_origin_sets(data_dir,data_type='test')
    
    # 得到特征选择id矩阵
    if k_best>0: select_dim,dim_to_fault = find_k_best(select_method=select_method,train_x=train_x,train_y=train_y,k_best=k_best)
    else: select_dim=[]
    
    # 预处理原始数据
    if preprocessing=='st': # Standardization（标准化）
        standard = StandardScaler() # 归一化
        train_x = standard.fit_transform(train_x)
        for i in range(fault):
            test_lx[i] = standard.transform(test_lx[i])
    if preprocessing=='mm': # MinMaxScaler (归一化)
        min_max_scaler = MinMaxScaler() # 归一化
        train_x = min_max_scaler.fit_transform(train_x)
        for i in range(fault):
            test_lx[i] = min_max_scaler.transform(test_lx[i])
            
    # 得到特征选择后的动态数据集
    if ocon:
        train_x,train_y = get_dynamic_datas(in_type='array', out_type='list', deal_type='no',
                                            X=train_x, select_dim=select_dim, dynamic=dynamic, one_hot=one_hot)
        test_lx,test_ly = get_dynamic_datas(in_type='list', out_type='list', deal_type='no',
                                            X=test_lx, select_dim=select_dim, dynamic=dynamic, one_hot=one_hot)
    else:
        train_x,train_y = get_dynamic_datas(in_type='array', out_type='array', deal_type='noise',
                                            X=train_x, select_dim=select_dim, dynamic=dynamic, one_hot=one_hot)
        test_lx,test_ly = get_dynamic_datas(in_type='list', out_type='list', deal_type='no',
                                            X=test_lx, select_dim=select_dim, dynamic=dynamic, one_hot=one_hot)
    
    # 对训练集洗牌    
    if shuffle: train_x,train_y = shuffle_data(train_x,train_y)
    
    # 返回train矩阵与test列表
    if ocon:    
        return train_x,train_y,test_lx,test_ly,dim_to_fault
    else:
        return train_x,train_y,test_lx,test_ly

dim=52
fault=22
n_train=480
n_test=800

#######################
#    data_analysis    #
#######################

def find_k_best(select_method='chi2',
                train_x=None,
                train_y=None,
                k_best=10):
    """
    功能：对各个fault选择最好的k个特征
    返回：特征选择id矩阵 x_id[fault,k_best]
    """
    # 特征选择方法
    if select_method=='chi2': m=chi2 # 卡方检验
    if select_method=='mi': m=mutual_info_classif # 互信息
    if select_method=='f': m=f_classif # 相关系数
    
    # 归一化
    min_max_scaler = MinMaxScaler() 
    train_x = min_max_scaler.fit_transform(train_x)
    
    # 得到用于特征选取的数据集(y=0 or 1)
    analysis_data=list()
    if train_y.shape[1]!=1:
        train_y=np.argmax(train_y,axis=1) # 不使用 one_hot 标签
    for i in range(22):
        y=np.zeros_like(train_y).astype(float)
        y[i*n_train:(i+1)*n_train]=1.
        analysis_data.append([train_x,y])

    # 对各fault选K_best
    x=[]
    x_id=[]
    for i,data_set in enumerate(analysis_data):
        print("select fault {}...".format(i))
        select_k_best = SelectKBest(m, k=k_best)
        se_x = select_k_best.fit_transform(data_set[0], data_set[1])
        se_id = select_k_best.get_support(indices=True).reshape(1,-1)
        se_x = se_x[i*n_train:(i+1)*n_train]
        if i==0:
            x=se_x
            x_id=se_id
        else: 
            x = np.concatenate((x,se_x),axis=0)
            x_id = np.concatenate((x_id,se_id),axis=0)  
    # 得到特征选择的 0 or 1矩阵
    dim_to_fault=np.zeros((dim,fault))
    for i in range(fault):
        se_for_fault = x_id[i].tolist()
        for j in range(dim):
            if j in se_for_fault: dim_to_fault[j][i]=1
            else: dim_to_fault[j][i]=0
        
    return x_id,dim_to_fault

if __name__ == "__main__":
    
    train_x , train_y , test_lx , test_ly , dim_to_fault = gene_net_datas(
            data_dir='../dataset/TE_csv',
            preprocessing='mm',
            one_hot=False, 
            shuffle=False,
            # 考虑动态数据集
            dynamic=10,
            # 用于特征选择
            select_method='chi2',
            k_best=10,
            # 一类一网络
            ocon=True)
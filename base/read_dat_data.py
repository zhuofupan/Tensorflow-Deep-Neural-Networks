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

def preprocess(train_x=None,test_lx=None,preprocessing=''):
    if preprocessing=='st': # Standardization（标准化）
        standard = StandardScaler() # 标准化
        train_x = standard.fit_transform(train_x)
        if test_lx is not None:
            for i in range(fault):
                test_lx[i] = standard.transform(test_lx[i])
    if preprocessing=='mm': # MinMaxScaler (归一化)
        min_max_scaler = MinMaxScaler() # 归一化
        train_x = min_max_scaler.fit_transform(train_x)
        if test_lx is not None:
            for i in range(fault):
                test_lx[i] = min_max_scaler.transform(test_lx[i])
    if test_lx is not None:
        return train_x,test_lx
    else:
        return train_x

def get_origin_sets(data_dir,
                    file_type='train',
                    data_type='array'):
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
        if _3_9_15==-1 and (i==3 or i==9 or i==15): continue ######################### 不考虑故障 3、9、15 #########################
        # 读取X
        if i<10: file_number='/d0'+str(i)
        else: file_number='/d'+str(i)
        if file_type=='train': 
            file=data_dir+file_number+'.dat'
            if i==0: data_x=np.transpose(np.loadtxt(file))
            else: data_x=np.loadtxt(file)
        else: 
            file=data_dir+file_number+'_te.dat'
            data_x=np.loadtxt(file)
        # 生成Y（非one-hot）
        n = data_x.shape[0]
        data_y = np.full(n,i).reshape(n,1)
        if file_type=='test':
            if d_start:
                data_x= np.delete(data_x,range(fault_start),axis=0)
                data_y= np.delete(data_y,range(fault_start),axis=0)
            else: data_y[:fault_start] = 0
        if _3_9_15==0 and (i==3 or i==9 or i==15): data_y=np.zeros_like(data_y) ############ 故障 3、9、15 视为 Normal ############
        data_y[:fault_start] = 0
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
        
    if data_type == 'array': return data_set_x,data_set_y
    else: return data_list_x,data_list_y
    
    
def get_dynamic_datas(file_type,
                      data_type,
                      X,
                      dynamic,
                      one_hot):
    """
    参数：dim=52，fault=22，n_train=480，n_test=960
    返回：train：得到特征选择后的动态数据集
         train_x[(480-dynamic+1)*22 , dynamic*52] , train_y[(480-dynamic+1)*22 , 22(1)]
         
         test：得到动态数据集列表
         test_x[22 , 800-dynamic+1 , dynamic*52] , train_y[22, 800-dynamic+1 , 22(1)]
    """
    def dynamic_data(x,label):
        # 动态数据集X
        dy_x=[]
        for r in range(x.shape[0]):
            if r+1>=dynamic:
                start=r+1-dynamic
                end=r+1
                nl=x[start:end].reshape(1,-1)
                if r+1==dynamic: dy_x=nl
                else: dy_x = np.concatenate((dy_x,nl),axis=0)
        dy_x=np.asarray(dy_x,dtype='float32')
        # 动态数据集Y
        n=dy_x.shape[0]
        if one_hot:
            dy_y = np.fromfunction(lambda j,k:(k==label).astype(int),(n,fault)) #.astype(int)
        else:
            dy_y = np.full(n,label).reshape(n,1)
        return dy_x,dy_y
    
    if data_type[1]=='array':
        data_x=[]
        data_y=[]
    else: 
        data_x=list()
        data_y=list()
    k=0
    for i in range(22):
        if _3_9_15==-1 and (i==3 or i==9 or i==15): continue
        # 原始数据
        if data_type[0]=='array':
            if file_type=='train':
                if k==0: file=X[:500,:]
                else: file=X[500+(k-1)*n_train:500+k*n_train,:]
            else:
                file=X[k*n_test:(k+1)*n_test,:]
        else: file=X[k]
        
        label=k
        if _3_9_15==0 and (i==3 or i==9 or i==15): label=0  ##################### 故障 3、9、15 视为 Normal #####################      
        if file_type=='train':
            dy_x,dy_y=dynamic_data(file,label)
        else:
            if d_start:
                dy_x,dy_y=dynamic_data(file,label)
            else:
                dy_x1,dy_y1=dynamic_data(file[:fault_start],0)
                dy_x2,dy_y2=dynamic_data(file[fault_start:],label)
                dy_x=np.concatenate((dy_x1,dy_x2),axis=0)
                dy_y=np.concatenate((dy_y1,dy_y2),axis=0)
        k=k+1
        # data_x
        if data_type[1]=='array':
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
                   k_best=0):
    """
    返回：train_x,train_y 矩阵（OCON=Flase）/列表（OCON=True）
         test_lx,test_ly 列表
         select_mat 对各个fault选取 k_best*dynamic 个特征，组成的[dim*dynamic,fault] 0-1 矩阵
    """
    # 得到原始数据集
    print('>>> Read data...')
    train_x,train_y = get_origin_sets(data_dir,file_type='train',data_type='array')
    test_lx,test_ly = get_origin_sets(data_dir,file_type='test',data_type='list')
    
    # 得到特征选择0-1矩阵
    select_01=[]
    if k_best!=0:
        print(">>> Feature selection...")
        if k_best>0:
            select_01 = find_k_best(train_x=train_x, train_y=train_y,
                                    select_method=select_method, k_best=k_best)
        else:
            select_01 = np.ones(52)
            select_01[22:41] = 0
        # 特征删减
        print(">>> Delete features...")
        train_x=fit_k_best(data_type='array', data_x=train_x, select_dim=select_01)
        test_lx=fit_k_best(data_type='list', data_x=test_lx, select_dim=select_01)
    
    # 预处理数据
    print(">>> Preprocessing...")
    train_x,test_lx = preprocess(train_x=train_x,test_lx=test_lx,preprocessing=preprocessing)
    
    # 得到动态数据集
    print('>>> Create dynamic data...')
    train_x,train_y = get_dynamic_datas(file_type='train',data_type=['array','array'], X=train_x, dynamic=dynamic, one_hot=one_hot)
    test_lx,test_ly = get_dynamic_datas(file_type='test',data_type=['list','list'], X=test_lx, dynamic=dynamic, one_hot=one_hot)
    
    # 对训练集洗牌    
    if shuffle:
        print(">>> Shuffle train data...")
        train_x,train_y = shuffle_data(train_x,train_y)
    
    # 返回train矩阵与test列表
    return train_x,train_y,test_lx,test_ly,select_01

dim=52
fault=22
n_train=480 # normal:500 , fault:480
n_test=960 # 960=160(normal)+800(fault)
fault_start=160
_3_9_15=-1 # 0 : 3、9、15 视为 Normal ; -1 : 不考虑 3、9、15
d_start=True # 删除测试集的前 fault_start 个
if _3_9_15==-1: fault=fault-3

#######################
#    data_analysis    #
#######################

def find_k_best(train_x=None,
                train_y=None,
                select_method='chi2',
                k_best=10):
    """
    功能：对各个(或所有)fault选择最好的k个特征
    返回：特征选择0-1矩阵 select_mat[dim*dynamic,fault]
    """
    # 特征选择方法
    if select_method=='chi2': m=chi2 # 卡方检验
    if select_method=='mi': m=mutual_info_classif # 互信息
    if select_method=='f': m=f_classif # 相关系数
    
    # 归一化
    train_x = preprocess(train_x=train_x,preprocessing='mm')
    # 使用 1位标签(不使用one-hot)
    if train_y.shape[1]!=1:
            train_y=np.argmax(train_y,axis=1)
    # 打乱
    train_x,train_y = shuffle_data(train_x,train_y)
    # 选择
    x_id=[]
    X=train_x
    Y=train_y
    Y=Y.reshape((Y.shape[0],))
    select_k_best = SelectKBest(m, k=k_best)
    select_k_best.fit_transform(X, Y) # 返回筛取k个特征后的X
    x_id = select_k_best.get_support(indices=False) # 返回特征选择与否的向量 or 选取的k个特征的序号向量
    select_dim=np.asarray(x_id,dtype='int32')
    return select_dim

def fit_k_best(data_type,data_x,select_dim):
    not_se_dim=list()
    for i in range(len(select_dim)):
        if select_dim[i]==0: not_se_dim.append(i)
    not_se_dim=np.asarray(not_se_dim).reshape((1,-1))
    if data_type=='array':
        train_x=np.delete(data_x,not_se_dim,axis=1)
        return train_x
    else:
        test_lx=list()
        for x in data_x:
            test_lx.append(np.delete(x,not_se_dim,axis=1))
        return test_lx

if __name__ == "__main__":
    
    train_x , train_y , test_lx , test_ly, select_mat = gene_net_datas(
            data_dir='../dataset/TE_dat',
            preprocessing='mm',
            one_hot=True, 
            shuffle=False,
            # 考虑动态数据集
            dynamic=9,
            # 用于特征选择
            select_method='f',
            k_best=33)
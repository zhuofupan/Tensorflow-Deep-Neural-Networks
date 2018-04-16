# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

m= 52
f= 22
r_f = [j/f for j in range(f)]
g_f = np.random.rand(f)
b_f = [1-j/f for j in range(f)]
r_m = [j/m for j in range(m)]
g_m = np.random.rand(m)
b_m = [1-j/m for j in range(m)]

def preprocess(train_lx=None,test_lx=None):
    train_x = []
    for i in range(len(train_lx)):
        if i==0: train_x=train_lx[i]
        else: train_x=np.concatenate((train_x,train_lx[i]),axis=0)
    min_max_scaler = MinMaxScaler() # 归一化
    train_x = min_max_scaler.fit_transform(train_x)
    train_lx=list()
    for i in range(f):
        if i==0: train_lx.append(train_x[:500])
        else: train_lx.append(train_x[500+(i-1)*480:500+i*480])
    for i in range(f):
        test_lx[i] = min_max_scaler.transform(test_lx[i])
    return train_lx,test_lx


def plot_origin(save='_te',tp='fault',X=None):
    if tp=='dim':
        # m*[x_m(t)*f]
        for i in range(m):
            print('>>> Plot X'+str(i)+save+'.png')
            fig = plt.figure(figsize=(45,25))
            for j in range(f):
                n=X[j].shape[0] # 500/480/960
                x=list(range(n))
                y=X[j][i]
                plt.plot(x, y,color=(r_f[j],g_f[j],b_f[j]),label='Fault'+str(j))
            plt.xlabel("$t$", fontsize=30)
            plt.ylabel("$x$", fontsize=30)
            plt.title("x_{}".format(i),fontsize=45)
            plt.legend(loc='upper right',fontsize=25)
            if not os.path.exists('plot/'): os.makedirs('plot/')
            plt.savefig('plot/X'+str(i)+save+'.png',bbox_inches='tight')
            plt.close(fig)
    else:
        # 22*[480,52]
        for i in range(f):
            n=X[i].shape[0] # 500/480/960
            x=list(range(n))
            y=X[i]
            print('>>> Plot Fault'+str(i)+save+'.png')
            fig = plt.figure(figsize=(45,25))
            for j in range(m):
                plt.plot(x, y[:,j],color=(r_m[j],g_m[j],b_m[j]),label='x'+str(j))
            plt.xlabel("$t$", fontsize=30)
            plt.ylabel("$x$", fontsize=30)
            plt.title("Fault{}".format(i),fontsize=45)
            plt.legend(loc='upper right',fontsize=25)
            if not os.path.exists('plot/'): os.makedirs('plot/')
            plt.savefig('plot/Fault'+str(i)+save+'.png',bbox_inches='tight')
            plt.close(fig)
            
from read_dat_data import get_origin_sets
print('>>> Read data...')
train_lx,_= get_origin_sets('../dataset/TE_dat',file_type='train',data_type='list')
test_lx,_= get_origin_sets('../dataset/TE_dat',file_type='test',data_type='list')
print(">>> Preprocessing...")
train_lx,test_lx = preprocess(train_lx=train_lx,test_lx=test_lx)

plot_origin(save='',tp='fault',X=train_lx)
plot_origin(save='_te',tp='fault',X=test_lx)

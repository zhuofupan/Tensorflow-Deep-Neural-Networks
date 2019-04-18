# -*- coding: utf-8 -*-
import tensorflow as tf

import numpy as np
np.random.seed(1337)

import sys
import os
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
from sup_sae import supervised_sAE
from read_usc_data import submission,read_data
from base_func import run_sess
    
datasets = read_data(meth='mfcc') 

#X_train, Y_train, X_test, Y_test =read_sf_data(dynamic=t)

x_dim=datasets[0].shape[1] # 52*dynamic
y_dim=datasets[1].shape[1] # 22(19) or 1

def run_(method=1,beta=None):
    tf.reset_default_graph() 
    # Training
    if method==1:
        classifier = DBN(
                     hidden_act_func='gauss',
                     output_act_func='gauss',
                     loss_func='mse', # gauss 激活函数会自动转换为 mse 损失函数
                     struct=[x_dim, y_dim*20, y_dim*10, y_dim],
                     # struct=[x_dim, int(x_dim/2), int(x_dim/4), int(x_dim/8), y_dim],
                     lr=1e-4,
                     use_for='classification',
                     bp_algorithm='rmsp',
                     epochs=240,
                     batch_size=16,
                     dropout=0.05,
                     units_type=['gauss','gauss'],
                     rbm_lr=1e-4,
                     rbm_epochs=45,
                     cd_k=1,
                     pre_train=True)
    elif method==2:
        classifier = supervised_sAE(
                     output_func='gauss',
                     hidden_func='gauss', 
                     loss_func='mse', 
                     struct=[x_dim, y_dim*60, y_dim*30, y_dim*10, y_dim],
                     lr=1e-4,
                     use_for='classification',
                     epochs=240,
                     batch_size=32,
                     dropout=0.34,
                     ae_type='yae', # ae | dae | sae
                     act_type=['gauss','affine'],# decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                     noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                     beta=beta, # DAE：噪声损失系数 | SAE：稀疏损失系数 | YAE：Y系数比重
                     p=0.3, # DAE：样本该维作为噪声的概率 | SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                     ae_lr=1e-4,
                     ae_epochs=30,
                     pre_train=True)
  
    run_sess(classifier,datasets,filename,load_saver='')
    return classifier

def repeat_(): # loop: beta = 0.1 → 1
    acc_list=np.zeros((y_dim+1,10))
    for i in range(10):
        beta = (i+1)/10
        classifier = run_(3,beta)
        for j in range(y_dim):
            acc_list[j][i] = classifier.acc_list[j]
        acc_list[y_dim][i] = np.mean(classifier.acc_list)
    return acc_list

classifier = run_(method=2,beta=1)
pred_class=classifier.pred_class
exp_time=classifier.pre_exp_time
submission(pred_class)
#acc_list = repeat_()

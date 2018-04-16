# -*- coding: utf-8 -*-
import tensorflow as tf
tf.reset_default_graph()

import numpy as np
np.random.seed(1337)

import sys
import os
sys.path.append("../models")
sys.path.append("../base")
from dbn import DBN
from sup_sae import supervised_sAE
from read_dat_data import gene_net_datas
from base_func import Initializer,Summaries

dbn_case=1

if dbn_case==1:
    units='bin'
    prepro='mm'
    class_func='softmax'
    r=1e-3
else:
    units='gauss'
    prepro='st'
    class_func='softmax'
    r=1e-5

X_train, Y_train, X_test, Y_test , select_mat = gene_net_datas(
        data_dir='../dataset/TE_dat',
        preprocessing=prepro, # gauss单元用‘st’, bin单元用'mm'
        # 考虑动态数据集
        dynamic=1,
        # 用于特征选择
        select_method='chi2', 
        k_best=0)
x_dim=X_train.shape[1] # 52*dynamic
y_dim=Y_train.shape[1] # 22(19) or 1

sae_case=1

if sae_case==1:
    ae_tp='ae'
    struct=[x_dim, 20, 10, y_dim]
    beta=0
    p=0
elif sae_case==2:
    ae_tp='sae'
    struct=[x_dim, 800, 300, y_dim]
    beta=0.25
    p=0.01
else:
    ae_tp='dae'
    struct=[x_dim, 300, 300, y_dim]
    beta=0.75
    p=0.2

sess = tf.Session()
# Training
method=2
if method==1:
    classifier = DBN(
                 output_act_func=class_func,
                 hidden_act_func='relu',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 use_for='classification',
                 bp_algorithm='mmt',
                 dbn_lr=r,
                 momentum=0.5,
                 dbn_epochs=100,
                 dbn_struct=[x_dim, y_dim*20, y_dim*10, y_dim],
                 rbm_v_type=units,
                 rbm_epochs=30,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3,
                 dropout=1)
else:
    classifier = supervised_sAE(
                 out_func='softmax',
                 en_func='affine', # encoder：[sigmoid] | [affine] 
                 use_for='classification',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 ae_type=ae_tp, # ae | dae | sae
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=beta,  # 惩罚因子权重（KL项 | 非噪声样本项）
                 p=p, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 sup_ae_struct=struct,
                 sup_ae_epochs=10,
                 ae_epochs=6,
                 batch_size=32,
                 ae_lr=1,
                 dropout=1)

Initializer.sess_init_all(sess) # 初始化变量
summ = Summaries(os.path.basename(__file__),sess=sess)
classifier.train_model(X_train, Y_train,sess=sess,summ=summ)

# Test
Y_pred=list()
print("Test data with Classifier...")
for i in range(y_dim):
    print(">>>Test fault {}:".format(i))
    Y_pred.append(classifier.test_model(X_test[i], Y_test[i],sess=sess))

sess.close()
summ.train_writer.close()
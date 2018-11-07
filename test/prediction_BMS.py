# -*- coding: utf-8 -*-
import tensorflow as tf

import numpy as np
np.random.seed(0)

import sys
import os
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
from sup_sae import supervised_sAE
from read_bms_data import preprocess,submission,read_data
    
train_X,train_Y,test_X = read_data() 

train_X = train_X[:,:8]
test_X = test_X [:,:8]
x_dim=train_X.shape[1] # 52*dynamic
y_dim=train_Y.shape[1] # 22(19) or 1
train_Y, _ ,prep = preprocess(train_Y,None,'mm')

def build_(method=1,beta=None):
    tf.reset_default_graph() 
    # Training
    if method==1:
        classifier = DBN(
                     hidden_act_func=['tanh','gauss'],
                     output_act_func='affine',
                     loss_func='mse', # gauss 激活函数会自动转换为 mse 损失函数
                     struct=[x_dim, x_dim*40, x_dim*20, x_dim*10, x_dim*2, y_dim],
                     lr=1e-4,
                     use_for='prediction',
                     bp_algorithm='rmsp',
                     epochs=240,
                     batch_size=32,
                     dropout=0.08,
                     units_type=['gauss','gauss'],
                     rbm_lr=1e-4,
                     rbm_epochs=60,
                     cd_k=1,
                     pre_train=True)
    elif method==2:
        classifier = supervised_sAE(
                     output_func='gauss',
                     hidden_func='affine', 
                     loss_func='mse', 
                     struct=[x_dim, x_dim*40, x_dim*20, x_dim*10, x_dim*2, y_dim],
                     lr=1e-4,
                     use_for='prediction',
                     epochs=180,
                     batch_size=32,
                     dropout=0.15,
                     ae_type='sae', # ae | dae | sae
                     act_type=['gauss','affine'],# decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                     noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                     beta=beta, # DAE：噪声损失系数 | SAE：稀疏损失系数 | YAE：Y系数比重
                     p=0.1, # DAE：样本该维作为噪声的概率 | SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                     ae_lr=1e-4,
                     ae_epochs=60,
                     pre_train=True)
    return classifier

classifier = build_(method=1,beta=0.7)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
classifier.train_model(train_X = train_X, 
                       train_Y = train_Y, 
                       sess = sess)
test_Y = sess.run(classifier.pred,feed_dict={
        classifier.input_data: test_X,
        classifier.keep_prob: 1.0})
test_Y = prep.inverse_transform(test_Y.reshape(-1,1))
exp_time=classifier.pre_exp_time
submission(test_Y)
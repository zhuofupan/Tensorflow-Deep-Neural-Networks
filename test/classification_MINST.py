import tensorflow as tf
tf.reset_default_graph()

import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import os
sys.path.append("../models")
sys.path.append("../base")
from dbn import DBN
from cnn import CNN
from sup_sae import supervised_sAE
from base_func import Initializer,Summaries
from tensorflow.examples.tutorials.mnist import input_data

# Loading dataset
# Each datapoint is a 8x8 image of a digit.
mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)

# Splitting data
X_train, Y_train, X_test, Y_test = mnist.train.images,mnist.train.labels,mnist.test.images , mnist.test.labels

#X_train, X_test = X_train[::100], X_test[::100]
#Y_train, Y_test = Y_train[::100], Y_test[::100]
x_dim=X_train.shape[1] 
y_dim=Y_train.shape[1] 
p_dim=int(np.sqrt(x_dim))

sess = tf.Session()
# Training
select_case = 2

if select_case==1:
    classifier = DBN(output_act_func='softmax',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='adag',
                 dbn_lr=1e-3,
                 momentum=0.5,
                 dbn_epochs=100,
                 dbn_struct=[x_dim, 200, 100, y_dim],
                 rbm_v_type='bin',
                 rbm_epochs=12,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3,
                 dropout=1)
if select_case==2:
    classifier = CNN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 cnn_lr=1e-3,
                 cnn_epochs=30,
                 img_shape=[p_dim,p_dim],
                 channels=[1, 6, 6, 64, y_dim], # 前几维给 ‘Conv’ ，后几维给 ‘Full connect’
                 layer_tp=['C','P','C','P'],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=1)
if select_case==3:
    classifier = supervised_sAE(
                 out_func='softmax',
                 en_func='affine',  # encoder：[sigmoid] | [affine] 
                 use_for='classification',
                 loss_func='mse',   # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 ae_type='ae',      # ae | dae | sae
                 noise_type='gs',   # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.6,          # 惩罚因子权重（KL项 | 非噪声样本项）
                 p=0.01,            # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 sup_ae_struct=[x_dim, 200, 50 , y_dim],
                 sup_ae_epochs=100,
                 ae_epochs=30,
                 batch_size=32,
                 ae_lr=1e-3,
                 dropout=1)

Initializer.sess_init_all(sess) # 初始化变量
summ = Summaries(os.path.basename(__file__),sess=sess)
classifier.train_model(X_train, Y_train,sess,summ)

# Test
print("[Test data...]")
Y_pred = classifier.test_model(X_test, Y_test,sess)

summ.train_writer.close()
sess.close()
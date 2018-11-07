import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import sys
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
from cnn import CNN
from sup_sae import supervised_sAE
from base_func import run_sess
from tensorflow.examples.tutorials.mnist import input_data

# Loading dataset
# Each datapoint is a 8x8 image of a digit.
mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)

# Splitting data
datasets = [mnist.train.images,mnist.train.labels,mnist.test.images , mnist.test.labels]

#X_train, X_test = X_train[::100], X_test[::100]
#Y_train, Y_test = Y_train[::100], Y_test[::100]
x_dim=datasets[0].shape[1] 
y_dim=datasets[1].shape[1] 
p_dim=int(np.sqrt(x_dim))

tf.reset_default_graph()
# Training
select_case = 3

if select_case==1:
    classifier = DBN(
                 hidden_act_func='sigmoid',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[x_dim, 400, 200, 100, y_dim],
                 lr=1e-3,
                 momentum=0.5,
                 use_for='classification',
                 bp_algorithm='rmsp',
                 epochs=100,
                 batch_size=32,
                 dropout=0.12,
                 units_type=['gauss','bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=16,
                 cd_k=1)
if select_case==2:
    classifier = CNN(
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 lr=1e-3,
                 epochs=30,
                 img_shape=[p_dim,p_dim],
                 channels=[1, 6, 6, 64, y_dim], # 前几维给 ‘Conv’ ，后几维给 ‘Full connect’
                 layer_tp=['C','P','C','P'],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=0.2)
if select_case==3:
    classifier = supervised_sAE(
                 output_func='softmax',
                 hidden_func='sigmoid', 
                 loss_func='cross_entropy', 
                 struct=[x_dim, 400, 200, 100, y_dim],
                 lr=1e-3,
                 use_for='classification',
                 epochs=100,
                 batch_size=32,
                 dropout=0.12,
                 ae_type='yae', # ae | dae | sae
                 act_type=['gauss','affine'],# decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5, # DAE：噪声损失系数 | SAE：稀疏损失系数
                 p=0.3, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 ae_lr=1e-3,
                 ae_epochs=16,
                 pre_train=True)
run_sess(classifier,datasets,filename,load_saver='')
label_distribution = classifier.label_distribution
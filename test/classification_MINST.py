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
select_case = 1

if select_case==1:
<<<<<<< HEAD
    classifier = DBN(
                 hidden_act_func='sigmoid',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[x_dim, 100, 50, y_dim],
                 lr=1e-3,
=======
    classifier = DBN(output_act_func='softmax',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='adam',
                 dbn_lr=1e-3,
>>>>>>> 4264eee5bcc3304abc64f7a1b22cf3c5b3cd37f4
                 momentum=0.5,
                 use_for='classification',
                 bp_algorithm='adam',
                 epochs=100,
                 batch_size=32,
                 dropout=0.3,
                 rbm_v_type='bin',
                 rbm_lr=1e-3,
                 rbm_epochs=24,
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

Initializer.sess_init_all(sess) # 初始化变量
summ = Summaries(os.path.basename(__file__),sess=sess)
classifier.train_model(train_X=X_train, train_Y=Y_train,sess=sess,summ=summ)

# Test
print("[Test data...]")
Y_pred = classifier.test_model(X_test, Y_test,sess)

summ.train_writer.close()
sess.close()

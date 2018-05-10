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
    classifier = DBN(
                 hidden_act_func='relu',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[x_dim, 200, 100, y_dim],
                 lr=1e-3,
                 momentum=0.5,
                 use_for='classification',
                 bp_algorithm='rmsp',
                 epochs=100,
                 batch_size=32,
                 dropout=0.1,
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

Initializer.sess_init_all(sess) # 初始化变量
summ = Summaries(os.path.basename(__file__),sess=sess)
classifier.train_model(train_X = X_train, 
                       train_Y = Y_train, 
                       val_X = X_test, 
                       val_Y = Y_test,
                       sess=sess,
                       summ=summ,
                       load_saver='')

# Test
print("[Test data...]")
print('[Average Accuracy]: %f' % classifier.best_acc)

label_distribution=classifier.label_distribution
record_array=classifier.record_array
np.savetxt("../saver/Label_distribution.csv", classifier.label_distribution, fmt='%.4f',delimiter=",")
np.savetxt("../saver/Loss_and_Acc.csv", classifier.record_array, fmt='%.4f',delimiter=",")

summ.train_writer.close()
sess.close()

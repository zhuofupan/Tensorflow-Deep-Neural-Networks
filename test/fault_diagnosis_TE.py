# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

np.random.seed(1337)
import sys
sys.path.append("../models")
sys.path.append("../base")
from dbn import DBN
from cnn import CNN
from read_te_data import gene_net_datas

dynamic=40
X_train, Y_train, X_test, Y_test = gene_net_datas(
        data_dir='../dataset/TE_csv',
        preprocessing='mm', # gauss单元用‘st’, bin单元用'mm'
        dynamic=dynamic)
# X_train.shape = [(480-dynamic+1)*22,dynamic*52]

dim=X_train.shape[1]
fault=Y_train.shape[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
select_case = 1

# gauss 激活函数会自动转换为 mse 损失函数
if select_case==1:
    classifier = DBN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 dbn_lr=1e-3,
                 dbn_epochs=100,
                 dbn_struct=[dim, 100, 100,fault],
                 rbm_v_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=10,
                 rbm_lr=1e-3,
                 dropout=0.95)
if select_case==2:
    classifier = CNN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 cnn_lr=1e-3,
                 cnn_epochs=100,
                 img_shape=[dynamic,52],
                 channels=[1, 6, 6, 64,fault],
                 fsize=[[4,4],[3,3]], 
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=0.9)

classifier.build_model()
classifier.train_model(X_train, Y_train,sess)

# Test
Y_pred=list()
print("[Test data...]")
for i in range(fault):
    print(">>>Test fault {}:".format(i))
    Y_pred.append(classifier.test_model(X_test[i], Y_test[i],sess))

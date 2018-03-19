# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

np.random.seed(1337)
import sys
sys.path.append("../models")
sys.path.append("../base")
from dbn import DBN
from read_te_data import read_data_sets
from sklearn.preprocessing import MinMaxScaler

# Splitting data
X_train, Y_train, X_test, Y_test = read_data_sets('../dataset/TE_csv',one_hot=True,shuffle=True)

min_max_scaler = MinMaxScaler() # 归一化
X_train = min_max_scaler.fit_transform(X_train)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
classifier = DBN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 dbn_lr=1e-3,
                 dbn_epochs=100,
                 dbn_struct=[52, 100, 100,22],
                 rbm_h_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3)

classifier.build_dbn()
classifier.train_dbn(X_train, Y_train,sess)

# Test
for i in range(len(X_test)):
    print(">>>Test fault {}:".format(i))
    X_test[i] = min_max_scaler.transform(X_test[i]) # 归一化
    Y_pred = classifier.test_dbn(X_test[i], Y_test[i],sess)

#import matplotlib.pyplot as plt
#PX=range(0,len(Y_pred))
#plt.figure(1)  # 选择图表1
#plt.plot(PX, Y_test,'r')
#plt.plot(PX, Y_pred,'b')
#plt.show()
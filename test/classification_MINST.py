import numpy as np
import tensorflow as tf

np.random.seed(1337)  # for reproducibility
import sys
sys.path.append("../models")
from dbn import DBN
from cnn import CNN
from tensorflow.examples.tutorials.mnist import input_data

# Loading dataset
# Each datapoint is a 8x8 image of a digit.
mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)

# Splitting data
X_train, X_test, Y_train, Y_test = mnist.train.images,mnist.test.images ,mnist.train.labels, mnist.test.labels

# X_train, X_test = X_train[::100], X_test[::100]
# Y_train, Y_test = Y_train[::100], Y_test[::100]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
select_case = 1

if select_case==1:
    classifier = DBN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 dbn_lr=1e-3,
                 dbn_epochs=100,
                 dbn_struct=[784, 100, 100,10],
                 rbm_v_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3)
if select_case==2:
    classifier = CNN(output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 cnn_lr=1e-3,
                 cnn_epochs=30,
                 img_shape=[28,28],
                 channels=[1, 6, 6, 64, 10],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=1)

classifier.build_model()
classifier.train_model(X_train, Y_train,sess)

# Test
print("[Test data...]")
Y_pred = classifier.test_model(X_test, Y_test,sess)

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from rbms import RBMs

class DBN(object):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 dbn_lr=1e-3,
                 dbn_epochs=100,
                 dbn_struct=[784, 100, 100,10],
                 rbm_h_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        self.output_act_func=output_act_func
        self.hidden_act_func=hidden_act_func
        self.loss_fuc=loss_fuc
        self.use_for=use_for
        self.dbn_lr=dbn_lr
        self.dbn_epochs=dbn_epochs
        self.dbn_struct = dbn_struct
        self.rbms_struct = dbn_struct[:-1]
        self.rbm_h_type=rbm_h_type
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        # 激活函数
        self.func_o=self.get_act_func(output_act_func)
        self.func_h=self.get_act_func(hidden_act_func)
        
    ###################
    #    DBN_model    #
    ###################
    
    def build_dbn(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.dbn_struct[0]]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.dbn_struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）
        # 权值 变量（初始化）
        self.out_W = tf.Variable(tf.truncated_normal(shape=[self.dbn_struct[-2], self.dbn_struct[-1]], stddev=0.1), name='out_W')
        self.out_b = tf.Variable(tf.constant(0.1, shape=[self.dbn_struct[-1]]),name='out_b')
        # 构建rbms
        self.rbms = RBMs(rbm_h_type=self.rbm_h_type,
                 rbms_struct=self.rbms_struct,
                 rbm_epochs=self.rbm_epochs,
                 batch_size=self.batch_size,
                 cd_k=self.cd_k,
                 rbm_lr=self.rbm_lr)
        self.rbms.build_rbms()
        # 构建dbn
        # 构建权值列表（dbn结构）
        self.parameter_list = list()
        for rbm in self.rbms.rbm_list:
            self.parameter_list.append(rbm.parameter)
        self.parameter_list.append([self.out_W,self.out_b])
        # 损失函数
        self.pred=self.prediction(self.input_data)
        self.loss=self.get_loss_func(self.loss_fuc)
        self.train_batch_bp=tf.train.AdamOptimizer(learning_rate=self.dbn_lr).minimize(self.loss, var_list=self.parameter_list)
        
    def train_dbn(self,train_X,train_Y,sess):
        sess.run(tf.global_variables_initializer())
        # 预训练
        print("[Start Pre-training...]")
        self.rbms.train_rbms(train_X,sess)
        # 微调
        print("[Start Training...]")
        # 参数设置
        self.images=train_X
        self.labels=train_Y
        self._images = train_X
        self._labels = train_Y
        self._num_examples = train_X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # 迭代次数
        for i in range(self.dbn_epochs):
            for _ in range(int(self._num_examples/self.batch_size)): 
                batch_x, batch_y= self.next_batch()
                loss,_=sess.run([self.loss,self.train_batch_bp],feed_dict={self.input_data: batch_x,self.label_data: batch_y})
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_dbn(self,test_X,test_Y,sess):
        print("[Test data...]")
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy(),self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[Accuracy]: %f' % acc)
            return pred_y
        else:
            loss,pred_y=sess.run([self.loss,self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[MSE]: %f' % loss)
            return pred_y
    
    def prediction(self,data_x):
        # 得到网络输出值
        next_data = data_x # 这个next_data是tf变量
        for parameter in self.parameter_list:
            W=parameter[0]
            b=parameter[1]
            z = tf.add(tf.matmul(next_data, W), b)
            if parameter==self.parameter_list[-1]:
                next_data=self.func_o(z)
            else:
                next_data=self.func_h(z)
        return next_data
    
    def accuracy(self):
        if self.dbn_struct[-1]>1:
            pre_lables=tf.argmax(self.pred,axis=1)
            data_lables=tf.argmax(self.label_data,axis=1)
        else:
            pre_lables=tf.floor(self.pred+0.5)
            data_lables=tf.floor(self.label_data+0.5)
        return tf.reduce_mean(tf.cast(tf.equal(pre_lables,data_lables),tf.float32))
        
    ###################
    #     DBN_func    #
    ###################
    
    def get_act_func(self,func_name):
        if func_name=='sigmoid': # S(z) = 1/(1+exp(-z)) ∈ (0,1)
            return tf.nn.sigmoid
        if func_name=='softmax': # s(z) = S(z)/∑S(z) ∈ (0,1)
            return tf.nn.softmax
        if func_name=='relu':    # r(z) = max(0,z) ∈ (0,+inf)
            return tf.nn.relu
        
    def get_loss_func(self,func_name):
        if func_name=='cross_entropy':
            if self.output_act_func=='softmax':
                return tf.losses.softmax_cross_entropy(self.label_data, self.pred)
            if self.output_act_func=='sigmoid':
                return tf.losses.sigmoid_cross_entropy(self.label_data, self.pred)
        if func_name=='mse':
            return tf.losses.mean_squared_error(self.label_data, self.pred)
    
    def next_batch(self,shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + self.batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += self.batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
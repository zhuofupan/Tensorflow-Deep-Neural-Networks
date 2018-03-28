# -*- coding: utf-8 -*-
import tensorflow as tf
from rbms import RBMs
import sys
sys.path.append("../base")
from base_func import Batch,Activation,Loss,Accuracy

class DBN(object):
    def __init__(self,
                 output_act_func='softmax',
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
                 rbm_lr=1e-3,
                 dropout=1):
        self.output_act_func=output_act_func
        self.hidden_act_func=hidden_act_func
        self.loss_fuc=loss_fuc
        self.use_for=use_for
        self.dbn_lr=dbn_lr
        self.dbn_epochs=dbn_epochs
        self.dbn_struct = dbn_struct
        self.rbms_struct = dbn_struct[:-1]
        self.rbm_v_type=rbm_v_type
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        self.dropout = dropout
        
        # 激活函数
        _act=Activation()
        if output_act_func=='gauss':
            self.loss_fuc='mse'
            self.func_o=self.gauss_func
        else:
            self.func_o=_act.get_act_func(self.output_act_func)
        self.func_h=_act.get_act_func(self.hidden_act_func)
        
    ###################
    #    DBN_model    #
    ###################
    
    def build_model(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.dbn_struct[0]]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.dbn_struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）
        # 权值 变量（初始化）
        self.out_W = tf.Variable(tf.truncated_normal(shape=[self.dbn_struct[-2], self.dbn_struct[-1]], stddev=0.1), name='W_out')
        self.out_b = tf.Variable(tf.constant(0.1, shape=[self.dbn_struct[-1]]),name='b_out')

        # 构建rbms
        self.rbms = RBMs(rbm_v_type=self.rbm_v_type,
                 rbms_struct=self.rbms_struct,
                 rbm_epochs=self.rbm_epochs,
                 batch_size=self.batch_size,
                 cd_k=self.cd_k,
                 rbm_lr=self.rbm_lr)
        self.rbms.build_model()
        # 构建dbn
        # 构建权值列表（dbn结构）
        self.parameter_list = list()
        for rbm in self.rbms.rbm_list:
            self.parameter_list.append(rbm.parameter)
        self.parameter_list.append([self.out_W,self.out_b])
        # 损失函数
        self.pred=self.transform(self.input_data)
        _loss=Loss(label_data=self.label_data,
                 pred=self.pred,
                 output_act_func=self.output_act_func)
        self.loss=_loss.get_loss_func(self.loss_fuc)
        self.train_batch_bp=tf.train.AdamOptimizer(learning_rate=self.dbn_lr).minimize(self.loss, var_list=self.parameter_list)
        # 正确率
        _ac=Accuracy(label_data=self.label_data,
                 pred=self.pred)
        self.accuracy=_ac.accuracy()
        
    def train_model(self,train_X,train_Y,sess):
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        # 预训练
        print("[Start Pre-training...]")
        self.rbms.train_model(train_X,sess)
        # 微调
        print("[Start Fine-tuning...]")  
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        # 迭代次数
        for i in range(self.dbn_epochs):
            for _ in range(int(n/self.batch_size)): 
                batch_x, batch_y= _data.next_batch()
                loss,_=sess.run([self.loss,self.train_batch_bp],feed_dict={self.input_data: batch_x,self.label_data: batch_y})
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_model(self,test_X,test_Y,sess):
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy,self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[Accuracy]: %f' % acc)
            return pred_y
        else:
            loss,pred_y=sess.run([self.loss,self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[MSE]: %f' % loss)
            return pred_y
    
    def transform(self,data_x):
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
            next_data = tf.nn.dropout(next_data, self.dropout)
        return next_data
    
    def gauss_func(self,x):
        return 1-tf.exp(-tf.square(x))
# -*- coding: utf-8 -*-
import tensorflow as tf
from rbms import DBM
import sys
sys.path.append("../base")
from base_func import Batch,Loss,Accuracy,Optimization,act_func,Summaries

class DBN(object):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='mmt',
                 dbn_lr=1e-3,
                 momentum=0.5,
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
        self.loss_func=loss_func
        self.use_for=use_for
        self.bp_algorithm=bp_algorithm
        self.dbn_lr=dbn_lr
        self.momentum=momentum
        self.dbn_epochs=dbn_epochs
        self.dbn_struct = dbn_struct
        self.dbm_struct = dbn_struct[:-1]
        self.rbm_v_type=rbm_v_type
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        self.dropout = dropout
        
        if output_act_func=='gauss':
            self.loss_func='mse'
        self.hidden_act=act_func(self.hidden_act_func)
        self.output_act=act_func(self.output_act_func)
        
        self.build_model()
        
    ###################
    #    DBN_model    #
    ###################
    
    def build_model(self):
        """
        Pre-training
        """
        # 构建dbm
        self.dbm = DBM(rbm_v_type=self.rbm_v_type,
                 dbm_struct=self.dbm_struct,
                 rbm_epochs=self.rbm_epochs,
                 batch_size=self.batch_size,
                 cd_k=self.cd_k,
                 rbm_lr=self.rbm_lr)
        
        """
        Fine-tuning
        """
        with tf.name_scope('DBN'):
            # feed 变量
            self.input_data = tf.placeholder(tf.float32, [None, self.dbn_struct[0]]) # N等于batch_size（训练）或_num_examples（测试）
            self.label_data = tf.placeholder(tf.float32, [None, self.dbn_struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）
            # 权值 变量（初始化）
            self.out_W = tf.Variable(tf.truncated_normal(shape=[self.dbn_struct[-2], self.dbn_struct[-1]], stddev=0.1), name='W_out')
            self.out_b = tf.Variable(tf.constant(0.1, shape=[self.dbn_struct[-1]]),name='b_out')
            # 构建dbn
            # 构建权值列表（dbn结构）
            self.parameter_list = list()
            for rbm in self.dbm.rbm_list:
                self.parameter_list.append(rbm.parameter)
            self.parameter_list.append([self.out_W,self.out_b])
            # 损失函数
            self.pred=self.transform(self.input_data)
            _loss=Loss(label_data=self.label_data,
                     pred=self.pred,
                     output_act_func=self.output_act_func)
            self.loss=_loss.get_loss_func(self.loss_func)
            _optimization=Optimization(r=self.dbn_lr,
                                       momentum=self.momentum)
            self.train_batch_bp=_optimization.trainer(algorithm=self.bp_algorithm).minimize(self.loss, var_list=self.parameter_list)
            # 正确率
            _ac=Accuracy(label_data=self.label_data,
                     pred=self.pred)
            self.accuracy=_ac.accuracy()
            
            #****************** 记录 ******************
            for i,parameter in enumerate(self.parameter_list):
                Summaries.scalars_histogram('_W'+str(i+1),parameter[0])
                Summaries.scalars_histogram('_b'+str(i+1),parameter[1])
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,tf.get_default_graph()._name_stack))
            #******************************************
        
    def train_model(self,train_X,train_Y,sess,summ):
        # 预训练
        print("Start Pre-training...")
        self.dbm.train_model(train_X,sess,summ)
        # 微调
        print("Start Fine-tuning...")  
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        m=int(n/self.batch_size)
        mod=max(int(self.rbm_epochs*m/1000),1)
        # 迭代次数
        k=0
        for i in range(self.dbn_epochs):
            for _ in range(m): 
                k=k+1
                batch_x, batch_y= _data.next_batch()
                summary,loss,_=sess.run([self.merge,self.loss,self.train_batch_bp],feed_dict={self.input_data: batch_x,self.label_data: batch_y})
                #**************** 写入 ******************
                if k%mod==0: summ.train_writer.add_summary(summary, k)
                #****************************************
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
                next_data=self.output_act(z)
            else:
                next_data=self.hidden_act(z)
            next_data = tf.nn.dropout(next_data, self.dropout)
        return next_data

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from model import Model
from base_func import act_func,out_act_check,Summaries

class RNN(Model):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='adam',
                 lr=1e-3,
                 momentum=0.9,
                 epochs=100,
                 width=5,
                 struct=[784, 100 ,10],
                 batch_size=32,
                 dropout=0):
        Model.__init__(self,'RNN')
        self.hidden_act_func=hidden_act_func
        self.output_act_func = out_act_check(output_act_func,loss_func)
        self.loss_func=loss_func
        self.use_for=use_for
        self.bp_algorithm=bp_algorithm
        self.lr=lr
        self.momentum=momentum
        self.epochs=epochs
        self.width=width
        self.struct = struct
        self.batch_size = batch_size
        self.dropout=dropout
            
        if output_act_func=='gauss':
            self.loss_func='mse'
        self.hidden_act=act_func(self.hidden_act_func)
        self.output_act=act_func(self.output_act_func)
        
        self.build_model()
        
    ###################
    #    RNN_model    #
    ###################
    
    def build_model(self):
        print("Start building model...")
        print('RNN:')
        print(self.__dict__)

        with tf.name_scope('RNN'):
            # feed 变量
            self.input_data = tf.placeholder(tf.float32, [None, self.struct[0]*self.width]) # N等于batch_size（训练）或_num_examples（测试）
            self.label_data = tf.placeholder(tf.float32, [None, self.struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）
            self.keep_prob = tf.placeholder(tf.float32) 
            X=tf.split(self.input_data,self.width,axis=1)
            
            self.create_variable()
            self.logits,self.pred = self.transform(X)
            self.build_train_step()
            
            #****************** 记录 ******************
            if self.tbd:
                for i in range(len(self.depth_parameter)):
                    Summaries.scalars_histogram('_W'+str(i+1),self.depth_parameter[i][0])
                    Summaries.scalars_histogram('_b'+str(i+1),self.depth_parameter[i][1])
                    if i < len(self.depth_parameter)-1:
                        Summaries.scalars_histogram('_V'+str(i+1),self.width_parameter[i])
                tf.summary.scalar('loss',self.loss)
                tf.summary.scalar('accuracy',self.accuracy)
                self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'RNN'))
            #******************************************
    
    def create_variable(self):
        # 定义所需变量
        self.depth_parameter=list()
        self.width_parameter=list()
        for i in range(len(self.struct)-1):
            W = tf.Variable(tf.truncated_normal(
                    shape=[self.struct[i], self.struct[i+1]], stddev=np.sqrt(2 / (self.struct[i] + self.struct[i+1]))
                    ), name='W')
            b = tf.Variable(tf.constant(0.0,shape=[self.struct[i+1]]),name='b')
            self.depth_parameter.append([W,b])
            if i<len(self.struct)-2:
                V = tf.Variable(tf.truncated_normal(
                            shape=[self.struct[i+1], self.struct[i+1]], stddev=np.sqrt(2 / (self.struct[i+1] + self.struct[i+1]))
                            ), name='V')
                self.width_parameter.append(V)
                
    def transform(self,X):
        # transform
        h_front=list()
        for i in range(self.width):
            with tf.name_scope('rnn-'+str(i+1)):
                next_data = X[i]
                for j in range(len(self.struct)-2):
                    W=self.depth_parameter[j][0]
                    b=self.depth_parameter[j][1]
                    V=self.width_parameter[j]
                    if self.dropout>0:
                        next_data = tf.nn.dropout(next_data, self.keep_prob)
                    z = tf.matmul(next_data,W) + b
                    if i==0:
                        next_data = self.hidden_act(z)
                        h_front.append(next_data)
                    else:
                        z = z + tf.matmul(h_front[j],V)
                        next_data = self.hidden_act(z)
                        h_front[j]=next_data
            if i==self.width-1:
                with tf.name_scope('classification'):
                    W=self.depth_parameter[-1][0]
                    b=self.depth_parameter[-1][1]
                    logits = tf.matmul(next_data,W) + b
                    pred = self.output_act(logits)
        return logits,pred
      
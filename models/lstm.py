# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from model import Model
from base_func import act_func,out_act_check,Summaries

class LSTM(Model):
    def __init__(self,
                 output_act_func='softmax',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='adam',
                 lr=1e-3,
                 momentum=0.9,
                 epochs=100,
                 width=5,
                 struct=[784, 100 ,10],
                 batch_size=32,
                 dropout=0,
                 variants=0):
        Model.__init__(self,'LSTM')
        self.loss_func=loss_func
        self.output_act_func = out_act_check(output_act_func,loss_func)
        self.use_for=use_for
        self.bp_algorithm=bp_algorithm
        self.lr=lr
        self.momentum=momentum
        self.epochs=epochs
        self.width=width
        self.struct = struct
        self.batch_size = batch_size
        self.dropout=dropout
        self.variants=variants
            
        if output_act_func=='gauss':
            self.loss_func='mse'
        self.output_act=act_func(self.output_act_func)
        
        self.build_model()
        
    ###################
    #    RNN_model    #
    ###################
    
    def build_model(self):
        print("Start building model...")
        print('LSTM:')
        print(self.__dict__)
        with tf.name_scope('LSTM'):
            # feed 变量
            self.input_data = tf.placeholder(tf.float32, [None, self.struct[0]*self.width]) # N等于batch_size（训练）或_num_examples（测试）
            self.label_data = tf.placeholder(tf.float32, [None, self.struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）
            self.keep_prob = tf.placeholder(tf.float32) 
            X=tf.split(self.input_data,self.width,axis=1)
            
            self.create_variable()
            
            # 构建训练步
            self.logits,self.pred = self.transform(X)
            self.build_train_step()
            
            #****************** 记录 ******************
            if self.tbd:
                for i in range(len(self.struct)-2):
                    Summaries.scalars_histogram('_F'+str(i+1),self.F_parameter[i][0])
                    Summaries.scalars_histogram('_I'+str(i+1),self.I_parameter[i][0])
                    Summaries.scalars_histogram('_C'+str(i+1),self.C_parameter[i][0])
                    Summaries.scalars_histogram('_O'+str(i+1),self.O_parameter[i][0])
                    Summaries.scalars_histogram('_bf'+str(i+1),self.F_parameter[i][1])
                    Summaries.scalars_histogram('_bi'+str(i+1),self.I_parameter[i][1])
                    Summaries.scalars_histogram('_bc'+str(i+1),self.C_parameter[i][1])
                    Summaries.scalars_histogram('_bo'+str(i+1),self.O_parameter[i][1])
                Summaries.scalars_histogram('_W',self.W)
                Summaries.scalars_histogram('_b',self.b)
                tf.summary.scalar('loss',self.loss)
                tf.summary.scalar('accuracy',self.accuracy)
                self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'LSTM'))
            #******************************************
    
    def create_variable(self):
        # 定义所需变量
        self.F_parameter=list()
        self.I_parameter=list()
        self.C_parameter=list()
        self.O_parameter=list()
        for i in range(len(self.struct)-1):
            n_x = self.struct[i]
            n_h = self.struct[i+1]         
            if self.variants==1:
                n_v = n_h + n_h + n_x
            else:
                n_v = n_h + n_x
            if i<len(self.struct)-2: # 前 N-1 层
                F = tf.Variable(tf.truncated_normal(shape=[n_v, n_h], stddev=np.sqrt(2 / (n_v + n_h))), name='F')
                I = tf.Variable(tf.truncated_normal(shape=[n_v, n_h], stddev=np.sqrt(2 / (n_v + n_h))), name='I')
                C = tf.Variable(tf.truncated_normal(shape=[n_v, n_h], stddev=np.sqrt(2 / (n_v + n_h))), name='C')
                O = tf.Variable(tf.truncated_normal(shape=[n_v, n_h], stddev=np.sqrt(2 / (n_v + n_h))), name='O')
                bf = tf.Variable(tf.constant(0.0,shape=[n_h]),name='bf')
                bi = tf.Variable(tf.constant(0.0,shape=[n_h]),name='bi')
                bc = tf.Variable(tf.constant(0.0,shape=[n_h]),name='bc')
                bo = tf.Variable(tf.constant(0.0,shape=[n_h]),name='bo')
                self.F_parameter.append([F,bf])
                self.I_parameter.append([I,bi])
                self.C_parameter.append([C,bc])
                self.O_parameter.append([O,bo])
            else: # 输出层
                self.W = tf.Variable(tf.truncated_normal(shape=[n_x, n_h], stddev=np.sqrt(2 / (n_x + n_h))), name='W')
                self.b = tf.Variable(tf.constant(0.0,shape=[n_h]),name='b')
                
    def next_data(self,x,k,j): # 第k(n_rnn)个第j(n_hidden)层
        F,bf=self.F_parameter[j][0],self.F_parameter[j][1]
        I,bi=self.I_parameter[j][0],self.I_parameter[j][1]
        C,bc=self.C_parameter[j][0],self.C_parameter[j][1]
        O,bo=self.O_parameter[j][0],self.O_parameter[j][1]
        if k ==0:
            n_x = self.struct[j]
            F=F[-1*n_x:]
            I=I[-1*n_x:]
            C=C[-1*n_x:]
            O=O[-1*n_x:]
            c_f=0
            h_f=0
            x_in = x
        else:
            c_f=self.c_front[j]
            h_f=self.h_front[j]
            if self.variants==1:
                x_in = tf.concat([c_f,h_f,x],1)
            else:
                x_in = tf.concat([h_f,x],1)    
        if self.dropout>0:
            x_in = tf.nn.dropout(x_in, self.keep_prob)
            
        f = tf.nn.sigmoid(tf.matmul(x_in,F) + bf)
        i = tf.nn.sigmoid(tf.matmul(x_in,I) + bi)
        if self.variants==0:
            c = tf.nn.tanh(tf.matmul(x_in,C) + bc)
            c = f * c_f + i * c
            o = tf.nn.sigmoid(tf.matmul(x_in,O) + bo)
            h = o * tf.nn.tanh(c)
        elif self.variants==1:
            c = tf.nn.tanh(tf.matmul(x_in,C) + bc)
            c = f * c_f + i * c
            if k==0:
                o_in = x
            else:
                o_in = tf.concat([c,h_f,x],1)
            o = tf.nn.sigmoid(tf.matmul(o_in,O) + bo)
            h = o * tf.nn.tanh(c)
        elif self.variants==2:
            c = tf.nn.tanh(tf.matmul(x_in,C) + bc)
            c = f * c_f + (1-f) * c
            o = tf.nn.sigmoid(tf.matmul(x_in,O) + bo)
            h = o * tf.nn.tanh(c)
        elif self.variants==3:
            if k==0:
                c_in = x
            else:
                c_in = tf.concat([i*h_f,x],1)
            c = tf.nn.tanh(tf.matmul(c_in,C) + bo)
            h = (1-f) * h_f + f * c
        if k==0:
            self.c_front.append(c)
            self.h_front.append(h)
        else:
            self.c_front[j]=c
            self.h_front[j]=h
        return h
                
    def transform(self,X):
        
        # transform
        self.h_front=list()
        self.c_front=list()
        for k in range(self.width):
            with tf.name_scope('lstm-'+str(k+1)):
                x = X[k]
                for j in range(len(self.struct)-2):
                    x=self.next_data(x,k,j)       
            if k==self.width-1:
                with tf.name_scope('classification'):
                    logits = tf.matmul(x,self.W) + self.b
                    pred = self.output_act(logits)
        return logits,pred

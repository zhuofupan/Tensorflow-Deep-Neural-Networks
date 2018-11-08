# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.append("../base")
from model import Model
from base_func import act_func,out_act_check,Summaries

class CNN(Model):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 lr=1e-3,
                 epochs=100,
                 img_shape=[28,28],
                 layer_tp=['C','C','P','C','P'],
                 channels=[1, 6, 6, 64, 10], # 输入1张图 -> 卷积池化成6张图 -> 卷积池化成6张图 -> 全连接层 -> 分类层
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=0):
        Model.__init__(self,'CNN')
        self.hidden_act_func=hidden_act_func
        self.output_act_func = out_act_check(output_act_func,loss_func)
        self.loss_func=loss_func
        self.use_for=use_for
        self.lr=lr
        self.epochs=epochs
        self.channels = channels
        self.layer_tp=layer_tp
        self.img_shape = img_shape
        self.fsize = fsize
        self.ksize = ksize
        self.batch_size = batch_size
        self.dropout = dropout
        
        with tf.name_scope('CNN'):
            self.build_model()
        
    ###################
    #    CNN_model    #
    ###################
    
    def build_model(self):
        print("Start building model...")
        print('CNN:')
        print(self.__dict__)
        # feed
        self.input_data = tf.placeholder(tf.float32, [None, self.img_shape[0]*self.img_shape[1]*self.channels[0]]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.channels[-1]]) # N等于batch_size（训练）或_num_examples（测试）
        self.keep_prob = tf.placeholder(tf.float32) 
        # reshape X
        X = tf.reshape(self.input_data, shape=[-1,self.img_shape[0] , self.img_shape[1], self.channels[0]])
        
        # 构建训练步
        self.logits,self.pred = self.transform(X)
        self.build_train_step()
        
        # Tensorboard
        if self.tbd:
            for i in range(len(self.W)):
                Summaries.scalars_histogram('_W'+str(i+1),self.W[i])
                Summaries.scalars_histogram('_b'+str(i+1),self.b[i])
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'CNN'))
        
    def conv2d(self,img, w, b):
        """tf.nn.conv2d
        input = [batch, in_height, in_width, in_channels]
        strides=[filter_height, filter_width, in_channels, out_channels]
        """
        z=tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID') + b
        return act_func(self.hidden_act_func)(z)

    def max_pool(self,img, ksize):
        return tf.nn.max_pool(img, ksize=ksize, strides=[1, 1, 1, 1], padding='VALID')
    
    def transform(self,X):
        # conv,max_pool
        self.W=list()
        self.b=list()
        
        c=0
        p=0
        for layer in self.layer_tp:
            if layer=='C':
                # W,b:conv
                name_W='W_conv'+str(c+1)
                name_b='b_conv'+str(c+1)
                W_shape=[self.fsize[c][0],self.fsize[c][1],self.channels[c],self.channels[c+1]]
                self.W.append(tf.Variable(tf.random_normal(shape=W_shape, stddev=0.1), name=name_W))
                self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[c+1]]),name=name_b))

                # 卷积操作
                """transform < conv >"""
                X = self.conv2d(X, self.W[-1], self.b[-1])
                c=c+1
            else:
                k_shape=[1,self.ksize[p][0],self.ksize[p][1],1]
                
                # 池化操作
                """transform < max_pool >"""
                X = self.max_pool(X, ksize=k_shape)
                p=p+1
            
        # full_connect 全连接层
        shape=X.get_shape().as_list()
        full_size=shape[1]* shape[2]*shape[3]
        X = tf.reshape(X, shape=[-1,full_size])
        for i in range(c,len(self.channels)-1):
            if i==c: n1=full_size
            else: n1=self.channels[i]
            n2=self.channels[i+1]
            
            # W,b:full_connect
            name_W='W_full'+str(i+1)
            name_b='b_full'+str(i+1)
            self.W.append(tf.Variable(tf.random_normal(shape=[n1,n2], stddev=0.1), name=name_W))
            self.b.append(tf.Variable(tf.constant(0.1, shape=[n2]),name=name_b))
            
            if self.dropout>0:
                    X = tf.nn.dropout(X, self.keep_prob) # Apply Dropout
            # 全连接
            """transform < full_connect >"""
            z=tf.add(tf.matmul(X,self.W[-1]), self.b[-1])
            
            if i==len(self.channels)-2:
                logits = z
                pred = act_func(self.output_act_func)(z) # Relu activation
            else:
                X = act_func(self.hidden_act_func)(z) # Relu activation 
        return logits,pred
# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.append("../base")
from base_func import Batch,Activation,Loss,Accuracy

class CNN(object):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_fuc='cross_entropy',
                 use_for='classification',
                 cnn_lr=1e-3,
                 cnn_epochs=100,
                 img_shape=[28,28],
                 channels=[1, 6, 6, 64, 10],
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=1):
        self.output_act_func=output_act_func
        self.hidden_act_func=hidden_act_func
        self.loss_fuc=loss_fuc
        self.use_for=use_for
        self.cnn_lr=cnn_lr
        self.cnn_epochs=cnn_epochs
        self.channels = channels
        self.img_shape = img_shape
        self.fsize = fsize
        self.ksize = ksize
        self.batch_size = batch_size
        self.dropout = dropout
        # 激活函数
        _act=Activation()
        self.func_o=_act.get_act_func(output_act_func)
        self.func_h=_act.get_act_func(hidden_act_func)
    
    ###################
    #    CNN_model    #
    ###################
    
    def build_model(self):
        # feed
        self.input_data = tf.placeholder(tf.float32, [None, self.img_shape[0]*self.img_shape[1]]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.channels[-1]]) # N等于batch_size（训练）或_num_examples（测试）
        # reshape X
        X = tf.reshape(self.input_data, shape=[-1,self.img_shape[0] , self.img_shape[1], self.channels[0]])
        
        # conv,max_pool
        self.W=list()
        self.b=list()
        for i in range(len(self.channels)-3):
            str_w='W'+str(i)
            str_b='b'+str(i)
            w_shape=[self.fsize[i][0],self.fsize[i][1],self.channels[i],self.channels[i+1]]
            k_shape=[1,self.ksize[i][0],self.ksize[i][1],1]
            # W,b:conv
            self.W.append(tf.Variable(tf.random_normal(shape=w_shape, stddev=0.1), name=str_w))
            self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[i+1]]),name=str_b))
            """transform <- conv,max_pool"""
            conv = self.conv2d(X, self.W[i], self.b[i])
            pool = self.max_pool(conv, ksize=k_shape)
            X = tf.nn.dropout(pool, self.dropout)
            
        # full_connect
        shape=X.get_shape().as_list()
        full_size=shape[1]* shape[2]*shape[3]
        X = tf.reshape(X, shape=[-1,full_size])
        # W,b:full_connect
        self.W.append(tf.Variable(tf.random_normal(shape=[full_size,self.channels[-2]], stddev=0.1), name='W_full'))
        self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[-2]]),name='b_full'))
        """transform <- full_connect"""
        z=tf.add(tf.matmul(X,self.W[-1]), self.b[-1])
        dense = self.func_h(z) # Relu activation
        X = tf.nn.dropout(dense, self.dropout) # Apply Dropout
        
        # classification
        # W,b:classification
        self.W.append(tf.Variable(tf.random_normal(shape=[self.channels[-2],self.channels[-1]], stddev=0.1), name='W_out'))
        self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[-1]]),name='b_out'))
        """transform <- classification"""
        z=tf.add(tf.matmul(X,self.W[-1]), self.b[-1])
        self.pred = self.func_o(z)
        
        # loss,trainer
        self.parameter_list=[self.W,self.b]
        _loss=Loss(label_data=self.label_data,
                 pred=self.pred,
                 output_act_func=self.output_act_func)
        self.loss=_loss.get_loss_func(self.loss_fuc)
        self.train_batch_bp=tf.train.AdamOptimizer(learning_rate=self.cnn_lr).minimize(self.loss, var_list=self.parameter_list)
        
        # accuracy
        _ac=Accuracy(label_data=self.label_data,
                 pred=self.pred)
        self.accuracy=_ac.accuracy()
        
    def conv2d(self,img, w, b):
        """tf.nn.conv2d
        input = [batch, in_height, in_width, in_channels]
        strides=[filter_height, filter_width, in_channels, out_channels]
        """
        xW=tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
        z=tf.nn.bias_add(xW,b)
        return self.func_h(z)

    def max_pool(self,img, ksize):
        return tf.nn.max_pool(img, ksize=ksize, strides=[1, 1, 1, 1], padding='VALID')
    
    def train_model(self,train_X,train_Y,sess):
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        # 训练
        print("[Start Training...]")   
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        # 迭代次数
        for i in range(self.cnn_epochs):
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
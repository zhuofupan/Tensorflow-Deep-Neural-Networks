# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.append("../base")
from base_func import Batch,act_func,Loss,Accuracy,Summaries

class CNN(object):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 cnn_lr=1e-3,
                 cnn_epochs=100,
                 img_shape=[28,28],
                 channels=[1, 6, 6, 64, 10], # 输入1张图 -> 卷积池化成6张图 -> 卷积池化成6张图 -> 全连接层 -> 分类层
                 fsize=[[4,4],[3,3]],
                 ksize=[[2,2],[2,2]],
                 batch_size=32,
                 dropout=1):
        self.output_act_func=output_act_func
        self.hidden_act_func=hidden_act_func
        self.loss_func=loss_func
        self.use_for=use_for
        self.cnn_lr=cnn_lr
        self.cnn_epochs=cnn_epochs
        self.channels = channels
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
        # feed
        self.input_data = tf.placeholder(tf.float32, [None, self.img_shape[0]*self.img_shape[1]]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.channels[-1]]) # N等于batch_size（训练）或_num_examples（测试）
        # reshape X
        X = tf.reshape(self.input_data, shape=[-1,self.img_shape[0] , self.img_shape[1], self.channels[0]])
        
        # conv,max_pool
        self.W=list()
        self.b=list()
        for i in range(len(self.channels)-3):
            # W,b:conv
            str_w='W'+str(i+1)
            str_b='b'+str(i+1)
            w_shape=[self.fsize[i][0],self.fsize[i][1],self.channels[i],self.channels[i+1]]
            k_shape=[1,self.ksize[i][0],self.ksize[i][1],1]
            self.W.append(tf.Variable(tf.random_normal(shape=w_shape, stddev=0.1), name=str_w))
            self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[i+1]]),name=str_b))
            
            # Tensorboard
            Summaries.scalars_histogram('_W'+str(i+1),self.W[-1])
            Summaries.scalars_histogram('_b'+str(i+1),self.b[-1])
            
            # 卷积、池化操作
            """transform < conv, max_pool >"""
            conv = self.conv2d(X, self.W[i], self.b[i])
            pool = self.max_pool(conv, ksize=k_shape)
            X = tf.nn.dropout(pool, self.dropout)
            
        # full_connect 全连接层
        shape=X.get_shape().as_list()
        full_size=shape[1]* shape[2]*shape[3]
        X = tf.reshape(X, shape=[-1,full_size])
        # W,b:full_connect
        self.W.append(tf.Variable(tf.random_normal(shape=[full_size,self.channels[-2]], stddev=0.1), name='W_full'))
        self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[-2]]),name='b_full'))
        
        # Tensorboard
        Summaries.scalars_histogram('_W_full',self.W[-1])
        Summaries.scalars_histogram('_b_full',self.b[-1])
        
        """transform < full_connect >"""
        z=tf.add(tf.matmul(X,self.W[-1]), self.b[-1])
        dense = act_func(self.hidden_act_func)(z) # Relu activation
        X = tf.nn.dropout(dense, self.dropout) # Apply Dropout
        
        # classification
        # W,b:classification
        self.W.append(tf.Variable(tf.random_normal(shape=[self.channels[-2],self.channels[-1]], stddev=0.1), name='W_out'))
        self.b.append(tf.Variable(tf.constant(0.1, shape=[self.channels[-1]]),name='b_out'))
        
        # Tensorboard
        Summaries.scalars_histogram('_W_out',self.W[-1])
        Summaries.scalars_histogram('_b_out',self.b[-1])
        
        """transform < classification >"""
        z=tf.add(tf.matmul(X,self.W[-1]), self.b[-1])
        self.pred = act_func(self.output_act_func)(z)
        
        # loss,trainer
        self.parameter_list=list()
        for i in range(len(self.W)):
            self.parameter_list.append(self.W[i])
            self.parameter_list.append(self.b[i])
        _loss=Loss(label_data=self.label_data,
                 pred=self.pred,
                 output_act_func=self.output_act_func)
        self.loss=_loss.get_loss_func(self.loss_func)
        self.train_batch_bp=tf.train.AdamOptimizer(learning_rate=self.cnn_lr).minimize(self.loss, var_list=self.parameter_list)
        
        # accuracy
        _ac=Accuracy(label_data=self.label_data,
                 pred=self.pred)
        self.accuracy=_ac.accuracy()
        
        # Tensorboard
        tf.summary.scalar('loss',self.loss)
        tf.summary.scalar('accuracy',self.accuracy)
        self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,tf.get_default_graph()._name_stack))
        
    def conv2d(self,img, w, b):
        """tf.nn.conv2d
        input = [batch, in_height, in_width, in_channels]
        strides=[filter_height, filter_width, in_channels, out_channels]
        """
        xW=tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
        z=tf.nn.bias_add(xW,b)
        return act_func(self.hidden_act_func)(z)

    def max_pool(self,img, ksize):
        return tf.nn.max_pool(img, ksize=ksize, strides=[1, 1, 1, 1], padding='VALID')
    
    def train_model(self,train_X,train_Y,sess,summ):
        # 训练
        print("[Start Training...]")   
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        m=int(n/self.batch_size)
        mod=max(int(self.cnn_epochs*m/1000),1)
        # 迭代次数
        k=0
        for i in range(self.cnn_epochs):
            for _ in range(m):
                k=k+1
                batch_x, batch_y= _data.next_batch()
                summary,loss,_=sess.run([self.merge,self.loss,self.train_batch_bp],feed_dict={self.input_data: batch_x,self.label_data: batch_y})
                #**************** 写入 ******************
                if k%mod==0: summ.train_writer.add_summary(summary, k)
                #****************************************
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_model(self,test_X,test_Y,sess):
        self.dropout=1.0
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy,self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[Accuracy]: %f' % acc)
            return pred_y
        else:
            loss,pred_y=sess.run([self.loss,self.pred],feed_dict={self.input_data: test_X,self.label_data: test_Y})
            print('[MSE]: %f' % loss)
            return pred_y

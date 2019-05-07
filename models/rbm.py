# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from model import Model
from base_func import act_func,Summaries

class RBM(Model):
    def __init__(self,
                 name='rbm',
                 units_type=['gauss','bin'],
                 rbm_struct=[784,100],
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        Model.__init__(self,name)
        self.name=name
        self.units_type=units_type
        self.n_v = rbm_struct[0]
        self.n_h = rbm_struct[1]
        self.epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.lr = rbm_lr
        self.decay_lr= True
        
        def conditional_probability(name):
            if name=='bin': return act_func('sigmoid')
            elif name=='gauss': return act_func('affine')
        
        self.h_func=conditional_probability(self.units_type[1])
        self.v_func=conditional_probability(self.units_type[0])
            
        with tf.name_scope(self.name):
            self.build_model()
            
            
    ###################
    #    RBM_model    #
    ###################
    
    def build_model(self):
        print(self.name + ':')
        print(self.__dict__)
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_v],name='X') # N等于batch_size（训练）或_num_examples（测试）
        self.recon_data = tf.placeholder(tf.float32, [None, self.n_v],name='recon_X')
        # 权值 变量（初始化）
        """
        tf.truncated_normal(shape=[self.n_v, self.n_h], stddev = np.sqrt(2 / (self.n_v + self.n_h)))
        tf.random_uniform(shape=[self.n_v, self.n_h], stddev = np.sqrt(6 / (self.n_v + self.n_h)))        
        tf.glorot_uniform_initializer()
        """
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_v, self.n_h], 
                                                 stddev = np.sqrt(2 / (self.n_v + self.n_h))), 
                                                 name='W')
        self.bh = tf.Variable(tf.constant(0.0,shape=[self.n_h]),name='bh')
        self.bv = tf.Variable(tf.constant(0.0,shape=[self.n_v]),name='bv')
        with tf.name_scope('CD-k'):
            # v0,h0
            v0=self.input_data # v0
            h0,s_h0=self.transform(v0) # h0
            # vk,hk
            logits,vk=self.reconstruction(s_h0) # v1
            for k in range(self.cd_k-1):
                _,s_hk=self.transform(vk) # trans（sample）
                logits,vk=self.reconstruction(s_hk) # recon（compute）
            hk,_=self.transform(vk) # hk
            self.logits=logits
            self.pred = vk
            
            with tf.name_scope('Gradient_Descent'):
                # upd8
                positive=tf.matmul(tf.expand_dims(v0,-1), tf.expand_dims(h0,1))
                negative=tf.matmul(tf.expand_dims(vk,-1), tf.expand_dims(hk,1))
                
                grad_W= tf.reduce_mean(tf.subtract(positive, negative), 0)
                grad_bh= tf.reduce_mean(tf.subtract(h0, hk), 0) 
                grad_bv= tf.reduce_mean(tf.subtract(v0, vk), 0) 

                self.w_upd8 = self.W.assign_add(grad_W *self.lr)
                self.bh_upd8 = self.bh.assign_add(grad_bh *self.lr)
                self.bv_upd8 = self.bv.assign_add(grad_bv *self.lr) 
                # 构建训练步
                self.train_batch =  [self.w_upd8, self.bh_upd8, self.bv_upd8]
                
        self.build_train_step()
        
        #****************** Tensorboard ******************
        if self.tbd:
            Summaries.scalars_histogram('_W',self.W)
            Summaries.scalars_histogram('_bh',self.bh)
            Summaries.scalars_histogram('_bv',self.bv)
            tf.summary.scalar('loss', self.loss)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,self.name))
        #*************************************************

    def transform(self,v):
        z = tf.matmul(v, self.W) + self.bh
        prob_h=self.h_func(z) # compute
        if self.units_type[1]=='gauss':
            state_h= prob_h
        else:
            rand_mat = tf.random_uniform(shape=tf.shape(prob_h),minval=0,maxval=1)
            state_h= tf.to_float(rand_mat<prob_h,name='state_h') # sample
        return prob_h,state_h
    
    def reconstruction(self,h):
        logits = tf.matmul(h, tf.transpose(self.W)) + self.bv
        prob_v=self.v_func(logits)
        return logits,prob_v

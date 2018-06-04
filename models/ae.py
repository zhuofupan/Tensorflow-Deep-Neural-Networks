# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from model import Model
from base_func import Loss,act_func,Summaries

class AE(Model):
    
    def __init__(self,
                 name='AE-1',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 ae_type='ae', # ae | dae | sae
                 act_type=['sigmoid','affine'],
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5,  # DAE：噪声损失系数 | SAE：稀疏损失系数
                 p=0.5, # DAE：样本该维作为噪声的概率 | SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 ae_struct=[784,100],
                 ae_epochs=10,
                 batch_size=32,
                 ae_lr=1e-3):
        Model.__init__(self,name)
        self.name=name
        self.loss_func=loss_func
        self.ae_type = ae_type
        
        self.noise_type = noise_type
        self.beta = beta
        self.alpha=1-self.beta
        self.p = p
        self.n_x = ae_struct[0]
        self.n_h = ae_struct[1]
        self.epochs = ae_epochs
        self.batch_size = batch_size
        self.lr = ae_lr
        self.momentum= 0.5
        
        self.en_func=act_type[0]
        self.de_func=act_type[1]
        # loss: cross_entropy 要求 h 必须是0~1之间的数
        if loss_func=='cross_entropy' and act_type[1]!='softmax' and act_type[1]!='sigmoid':
            self.loss_func = 'mse'
        # sae: cross_entropy 要求 h 必须是0~1之间的数
        if ae_type=='sae' and act_type[1]!='softmax' and act_type[1]!='sigmoid':
            self.de_func = 'sigmoid'
        
        with tf.name_scope(self.name):
            self.build_model()
            
    ###################
    #    RBM_model    #
    ###################
    
    def build_model(self):
        print(self.name + ':')
        print(self.__dict__)
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_x],name='X') # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.n_x],name='recon_X')
        
        # 权值 变量（初始化）
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_x, self.n_h], 
                                                 stddev = np.sqrt(2 / (self.n_x + self.n_h))), 
                                                 name='W')
        self.bz = tf.Variable(tf.constant(0.0, shape=[self.n_x]),name='bz')
        self.bh = tf.Variable(tf.constant(0.0, shape=[self.n_h]),name='bh')
            
        self.var_list = [self.W, self.bh, self.bz]
        
        # 建模
        x=self.input_data
        if self.ae_type=='dae': # 去噪自编码器 [dae]
            x_=self.add_noise(x)
            h=self.transform(x_)
            self.logist,self.pred=self.reconstruction(h)
            self.loss = self.denoising_loss(x,self.pred)
        else: 
            h=self.transform(x) # 自编码器 [ae]
            self.logist,self.pred=self.reconstruction(h)
            _loss=Loss(label_data=self.input_data, 
                       pred=self.pred,
                       logist=self.logist,
                       output_act_func=self.de_func)
            self.loss=_loss.get_loss_func(self.loss_func)
            if self.ae_type=='sae': # 稀疏自编码器 [sae]
                self.loss = self.alpha * self.loss + self.beta * self.sparse_loss(h) 
        
        # 构建训练步
        self.build_train_step()
        
        #****************** Tensorboard ******************
        if self.tbd:
            Summaries.scalars_histogram('_W',self.W)
            Summaries.scalars_histogram('_bz',self.bz)
            Summaries.scalars_histogram('_bh',self.bh)
            tf.summary.scalar('loss', self.loss)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,self.name))
        #*************************************************

    def transform(self,x):
        add_in = tf.matmul(x, self.W) + self.bh
        return act_func(self.en_func)(add_in)
    
    def reconstruction(self,h):
        logist = tf.matmul(h, tf.transpose(self.W)) + self.bz
        recon = act_func(self.de_func)(logist)
        return logist,recon
    
    # DAE
    def add_noise(self,x):
        # A为损失系数矩阵，一行对应一个样本，引入噪声的维度系数为alpha，未引入噪声的为beta
        rand_mat = tf.random_uniform(shape=tf.shape(x),minval=0,maxval=1)
        self.A_ = tf.to_float(rand_mat<self.p,name='Noise') # 噪声系数矩阵
        self.A = 1-self.A_ # 保留系数矩阵
        if self.noise_type=='gs':
            rand_gauss = tf.truncated_normal(x.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
            x_ = x * self.A + rand_gauss * self.A_
        else:
            x_ = x * self.A
        return x_
    
    def denoising_loss(self,y,y_):
        if self.loss_func=='mse':
            loss_mat=tf.square(y-y_)
        elif self.loss_func=='cross_entropy':
            y = tf.clip_by_value(y,1e-10, 1.0-1e-10)
            y_ = tf.clip_by_value(y_,1e-10, 1.0-1e-10)
            if self.de_func=='sigmoid':
                loss_mat=-y*tf.log(y_)-(1-y)*tf.log(1-y_)
            elif self.de_func=='softmax':
                loss_mat=-y * tf.log(y_)
        loss = self.alpha * loss_mat * self.A + self.beta * loss_mat * self.A_
        if self.loss_func=='cross_entropy' and self.de_func=='softmax':
            loss_mat=tf.reduce_sum(loss_mat,axis=1)
        return tf.reduce_mean(loss)
    
    # SAE        
    def sparse_loss(self,h):
        q = tf.reduce_mean(h,axis=0)
        p = tf.constant(self.p, shape=[1,self.n_h])
        q = tf.clip_by_value(q,1e-10, 1.0-1e-10)
        p = tf.clip_by_value(p,1e-10, 1.0-1e-10)
        kl=p*tf.log(p/q)+(1-p)*tf.log((1-p)/(1-q))
        return tf.reduce_sum(kl)

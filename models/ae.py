# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from model import Model
from base_func import Loss,act_func,out_act_check,Summaries

class AE(Model):
    
    def __init__(self,
                 name='AE-1',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 ae_type='ae', # ae | dae | sae
                 act_type=['sigmoid','affine'],
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5,  # DAE：噪声损失系数 | SAE：稀疏损失系数
                 p=0.5, # DAE：样本该维作为噪声的概率 | SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 struct=[784,100],
                 out_size = 10,
                 ae_epochs=10,
                 batch_size=32,
                 lr=1e-3):
        Model.__init__(self,name)
        self.name=name
        self.loss_func=loss_func
        self.ae_type = ae_type
        
        self.noise_type = noise_type
        self.beta = beta
        self.alpha=1-self.beta
        self.p = p
        self.n_x = struct[0]
        self.n_h = struct[1]
        self.out_size = out_size
        self.epochs = ae_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum= 0.5
        
        self.en_func=act_type[0]
        self.de_func= out_act_check(act_type[1],loss_func)
        
        # sae: kl 要求 h 必须是0~1之间的数
        if ae_type=='sae' and (act_type[0] not in ['softmax','sigmoid','gauss']):
            self.en_func = 'sigmoid'
        
        with tf.name_scope(self.name):
            self.build_model()
            
    ###################
    #     AE_model    #
    ###################
    
    def build_model(self):
        print(self.name + ':')
        print(self.__dict__)
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_x],name='X') # N等于batch_size（训练）或_num_examples（测试）
        self.recon_data = tf.placeholder(tf.float32, [None, self.n_x],name='recon_X')
        
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
            self.logits,self.pred=self.reconstruction(h)
            _loss=Loss(label=self.recon_data, 
                       logits=self.logits,
                       out_func_name=self.de_func,
                       loss_name=self.loss_func)
            loss_mat,_=_loss.get_loss_mat()
            self.loss = self.denoising_loss(loss_mat)
        else: 
            h=self.transform(x) # 自编码器 [ae]
            self.logits,self.pred=self.reconstruction(h)
            # 这部分损失共用
            _loss=Loss(label=self.recon_data, 
                       logits=self.logits,
                       out_func_name=self.de_func,
                       loss_name=self.loss_func)
            self.loss=_loss.get_loss_func()
            #_,self.loss=_loss.get_loss_mat()
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
        logits = tf.matmul(h, tf.transpose(self.W)) + self.bz
        recon = act_func(self.de_func)(logits)
        return logits,recon
    
    ###############
    #     DAE     #
    ###############
    
    # 加噪声
    def add_noise(self,x):
        """
            A为损失系数矩阵，一行对应一个样本，引入噪声的变量的系数为alpha，未引入噪声的为beta
            当噪声类型为 Masking noise (mn) 时，相当于做 dropout
        """ 
        rand_mat = tf.random_uniform(shape=tf.shape(x),minval=0,maxval=1)
        self.A_ = tf.to_float(rand_mat<self.p,name='Noise') # 噪声系数矩阵
        self.A = 1-self.A_ # 保留系数矩阵
        if self.noise_type=='gs':
            rand_gauss = tf.truncated_normal(x.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
            x_ = x * self.A + rand_gauss * self.A_
        else:
            x_ = x * self.A
        return x_
    
    # 返回 loss 值 
    def denoising_loss(self,loss_mat):
        loss_mat = self.alpha * loss_mat * self.A + self.beta * loss_mat * self.A_
        if self.loss_func=='cross_entropy' and self.de_func=='softmax':
            loss_mat=tf.reduce_sum(loss_mat,axis=1)
        return tf.reduce_mean(loss_mat)
    
    ###############
    #     SAE     #
    ###############
       
    def sparse_loss(self,h):
        q = tf.clip_by_value(tf.reduce_mean(h,axis=0),1e-10, 1.0-1e-10)
        p = tf.clip_by_value(tf.constant(self.p, shape=[1,self.n_h]),1e-10, 1.0-1e-10)
        kl=p*tf.log(p/q)+(1-p)*tf.log((1-p)/(1-q))
        return tf.reduce_sum(kl)

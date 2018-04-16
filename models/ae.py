# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../base")
from base_func import Batch,Loss,Optimization,act_func,Summaries

class AE(object):
    
    def __init__(self,
                 name='AE-1',
                 en_func='sigmoid', # encoder：[sigmoid]
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 ae_type='ae', # ae | dae | sae
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5,  # 惩罚因子权重（第二项损失的系数）
                 p=0.5, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 ae_struct=[784,100],
                 ae_epochs=10,
                 batch_size=32,
                 ae_lr=1e-3):
        self.name=name
        self.en_func=en_func
        self.loss_func=loss_func
        self.ae_type = ae_type
        self.noise_type = noise_type
        self.beta = beta
        self.p = p
        self.n_x = ae_struct[0]
        self.n_y = ae_struct[1]
        self.ae_epochs = ae_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        self.momentum=0.0
        if loss_func=='mse': self.de_func='affine'
        else : self.de_func='sigmoid'
        if ae_type=='sae'and self.de_func=='affine': self.de_func='relu'
        
        with tf.name_scope(self.name):
            self.build_model()
        
    ###################
    #    RBM_model    #
    ###################
    
    def build_model(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_x],name='X') # N等于batch_size（训练）或_num_examples（测试）
        self.A = tf.placeholder(tf.float32, [None, self.n_x],name='A') 
        # 权值 变量（初始化）
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_x, self.n_y], stddev=0.1), name='W')
        self.bz = tf.Variable(tf.constant(0.1, shape=[self.n_x]),name='bz')
        self.by = tf.Variable(tf.constant(0.1, shape=[self.n_y]),name='by')
        self.p_mat = tf.Variable(tf.constant(self.p, shape=[1,self.n_y]),name='p_mat')
            
        self.var_list = [self.W, self.by, self.bz]
        
        # 建模
        x=self.input_data
        y=self.transform(x)
        z=self.reconstruction(y)
        if self.ae_type=='dae': # 去噪自编码器 [dae]
            self.loss = self.get_denoising_loss(x,z)
        else: 
            _loss=Loss(label_data=self.input_data, # 自编码器 [ae]
                       pred=z,
                       output_act_func=self.de_func)
            self.loss=_loss.get_loss_func(self.loss_func)
            if self.ae_type=='sae': # 稀疏自编码器 [sae]
                self.loss = (1-self.beta)*self.loss + self.beta*self.KL(y) 
        
        _optimization=Optimization(r=self.ae_lr,
                                   momentum=self.momentum,
                                   use_nesterov=True)
        self.train_batch_bp=_optimization.trainer(algorithm='sgd').minimize(self.loss, var_list=self.var_list)
        
        #****************** Tensorboard ******************
        Summaries.scalars_histogram('_W',self.W)
        Summaries.scalars_histogram('_bz',self.bz)
        Summaries.scalars_histogram('_by',self.by)
        tf.summary.scalar('loss', self.loss)
        self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,tf.get_default_graph()._name_stack))
        #*************************************************
        
    def train_model(self,train_X,sess,summ):
        # 初始化变量
        _data=Batch(images=train_X,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        m=int(n/self.batch_size)
        mod=max(int(self.ae_epochs*m/1000),1)
        # 迭代次数
        k=0
        A=np.ones((self.batch_size,self.n_x),dtype=np.float32)
        for i in range(self.ae_epochs):
            # 批次训练
            self.momentum=i/self.ae_epochs
            for _ in range(m): 
                k=k+1
                batch = _data.next_batch()
                if self.ae_type=='dae':
                    batch,A=self.add_noise(batch)
                summary,loss,_= sess.run([self.merge,self.loss,self.train_batch_bp],
                                         feed_dict={self.input_data: batch,self.A: A})
                
                #**************** 写入 ******************
                if k%mod==0: summ.train_writer.add_summary(summary, k)
                #****************************************   
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def add_noise(self,x):
        # A为损失系数矩阵，一行对应一个样本，引入噪声的维度系数为alpha，未引入噪声的为beta
        A=np.ones((x.shape[0],x.shape[1]))*self.beta
        self.alpha=1-self.beta
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.random.rand(1) < self.p:
                    if self.noise_type=='gs':
                        noise=np.random.standard_normal(1)
                    else:
                        noise=0.
                    A[i][j]=self.alpha
                    x[i][j]=noise
        return x,A
    
    def transform(self,x):
        add_in = tf.add(tf.matmul(x, self.W), self.by)
        return act_func(self.en_func)(add_in)
    
    def reconstruction(self,y):
        add_in = tf.add(tf.matmul(y, tf.transpose(self.W)), self.bz)
        if self.loss_func=='mse': return add_in
        else: return tf.nn.sigmoid(add_in)
        
    def samples(self,y):
        return tf.round(y) # 舍近采样
        # return tf.to_float(tf.random_uniform([tf.shape(y)[0],self.n_y])<y) # 随机采样
    
    def get_denoising_loss(self,x,z):
        if self.loss_func=='mse':
            loss_mat=tf.square(tf.subtract(x,z))
        elif self.loss_func=='cross_entropy':
            x = tf.clip_by_value(x, 1e-10, 1.0)
            z = tf.clip_by_value(z, 1e-10, 1.0)
            loss_mat=-x*tf.log(z)/tf.log(2.)-(1-x)*tf.log(1-z)/tf.log(2.)
        return tf.reduce_mean(tf.reduce_sum(loss_mat*self.A,axis=1))
            
    def KL(self,y):
        q = tf.clip_by_value(tf.reduce_mean(y,0), 1e-10, 1.0) # 平均激活度 [1,Y]
        p = tf.clip_by_value(self.p_mat, 1e-10, 1.0)
        kl=p*tf.log(p/q)/tf.log(2.)+(1-p)*tf.log((1-p)/(1-q))/tf.log(2.)
        return tf.reduce_sum(kl)

if __name__ == "__main__":
    ae=AE()
    ae.build_model()
    print('ok')
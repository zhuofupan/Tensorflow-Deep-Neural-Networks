# -*- coding: utf-8 -*-
import tensorflow as tf
from rbm import RBM

class DBM(object):
    def __init__(self,
                 rbm_v_type='bin',
                 dbm_struct=[784, 100, 100],
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        self.rbm_v_type=rbm_v_type
        self.dbm_struct = dbm_struct
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        
        self.build_model()
        
    ####################
    #    DBM_model    #
    ####################
    
    def build_model(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.dbm_struct[0]]) # N等于_num_examples或batch_size
        # 构建rmbs
        self.rbm_list = list()
        for i in range(len(self.dbm_struct) -1):
            n_v = self.dbm_struct[i]
            n_h = self.dbm_struct[i+1]
            name='rbm-'+ str(i + 1)
            rbm = RBM(name=name,
                      rbm_v_type=self.rbm_v_type,
                      rbm_struct=[n_v,n_h],
                      rbm_epochs=self.rbm_epochs,
                      batch_size=self.batch_size,
                      cd_k=self.cd_k,
                      rbm_lr=self.rbm_lr)
            self.rbm_list.append(rbm) # 加入list
            
    def train_model(self,train_X,sess,summ):
        next_data = train_X # 这个next_data是实数
        for i,rbm in enumerate(self.rbm_list):
            print('>>> Training RBM-{}:'.format(i+1))
            # 训练第i个RBM（按batch）
            rbm.train_model(next_data,sess,summ)
            # 得到transform值（train_X）
            next_data,_ = sess.run(rbm.transform(next_data))

# -*- coding: utf-8 -*-
import tensorflow as tf
from rbm import RBM

class DBM(object):
    def __init__(self,
                 dbm_struct=[784, 100, 100],
                 units_type=['gauss','bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1):
        self.units_type = units_type
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
        self.pt_list = list()
        for i in range(len(self.dbm_struct) -1):
            n_v = self.dbm_struct[i]
            n_h = self.dbm_struct[i+1]
            name='rbm-'+ str(i + 1)
            rbm = RBM(name=name,
                      units_type=self.units_type,
                      rbm_struct=[n_v,n_h],
                      rbm_epochs=self.rbm_epochs,
                      batch_size=self.batch_size,
                      cd_k=self.cd_k,
                      rbm_lr=self.rbm_lr)
            self.pt_list.append(rbm) # 加入list
            
    def train_model(self,train_X,sess,summ):
        X = train_X 
        for i,rbm in enumerate(self.pt_list):
            print('>>> Train RBM-{}:'.format(i+1))
            # 训练第i个RBM（按batch）
            rbm.unsupervised_train_model(train_X=X,sess=sess,summ=summ)
            # 得到transform值（train_X）
            X,_ = sess.run(rbm.transform(X))
            
    def test_model(self,test_X,sess):
        return sess.run(self.transform(self.input_data),feed_dict={self.input_data: test_X})
    
    def transform(self,data_x):
        next_data = data_x # 这个next_data是tf变量
        for rbm in self.pt_list:
            next_data,_ = rbm.transform(next_data)
        return next_data

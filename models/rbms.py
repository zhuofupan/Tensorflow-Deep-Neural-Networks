# -*- coding: utf-8 -*-
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
        # 构建rmbs
        self.pt_list = list()
        self.parameter_list=list()
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
            self.parameter_list.append([rbm.W,rbm.bh])
            
    def train_model(self,train_X,train_Y,sess,summ):
        # 返回最后一层特征<实值>
        X = train_X 
        for i,rbm in enumerate(self.pt_list):
            print('>>> Train RBM-{}:'.format(i+1))
            # 训练第i个RBM（按batch）
            rbm.unsupervised_train_model(train_X=X,train_Y=train_Y,sess=sess,summ=summ)
            # 得到transform值（train_X）
            X,_ = sess.run(rbm.transform(X))
        return X
    
    def transform(self,X):
        # 返回最后一层特征<tf变量>
        for rbm in self.pt_list:
            X = rbm.transform(X)
        return X
        
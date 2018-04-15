# -*- coding: utf-8 -*-
import tensorflow as tf
from dbn import DBN
import sys
sys.path.append("../base")
from base_func import Batch,Activation,Loss,Accuracy,Optimization

class OVON(object):
    def __init__(self,
                 output_act_func='softmax',
                 hidden_act_func='relu',
                 loss_func='cross_entropy',
                 use_for='classification',
                 bp_algorithm='mmt',
                 dbn_lr=1e-3,
                 momentum=0.5,
                 dbn_epochs=100,
                 dbn_struct=[40, 100, 10],
                 rbm_v_type='bin',
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3,
                 dropout=1):
        self.output_act_func=output_act_func
        self.hidden_act_func=hidden_act_func
        self.loss_func=loss_func
        self.use_for=use_for
        self.bp_algorithm=bp_algorithm
        self.dbn_lr=dbn_lr
        self.momentum=momentum
        self.dbn_epochs=dbn_epochs
        self.dbn_struct = dbn_struct
        self.dynamic = dbn_struct[0]
        self.rbm_v_type=rbm_v_type
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        self.dropout = dropout
        
        # 激活函数
        _act=Activation()
        if output_act_func=='gauss':
            self.loss_fuc='mse'
            self.func_o=self.gauss_func
        else:
            self.func_o=_act.get_act_func(self.output_act_func)
        self.func_h=_act.get_act_func(self.hidden_act_func)
        
    ###################
    #    DBN_model    #
    ###################
    
    def build_model(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, 52*self.dynamic]) # N等于batch_size（训练）或_num_examples（测试）
        self.label_data = tf.placeholder(tf.float32, [None, self.dbn_struct[-1]]) # N等于batch_size（训练）或_num_examples（测试）

        # 构建dbn_list
        self.dbn_list=list()
        for i in range(52):
            dbn = DBN(output_act_func=self.output_act_func,
                      hidden_act_func=self.hidden_act_func,
                      loss_fuc=self.loss_fuc,
                      use_for=self.use_for,
                      bp_algorithm=self.bp_algorithm,
                      dbn_lr=self.dbn_lr,
                      momentum=self.momentum,
                      dbn_epochs=self.dbn_epochs,
                      dbn_struct=self.dbn_struct,
                      rbm_v_type=self.rbm_v_type,
                      rbm_epochs=self.rbm_epochs,
                      batch_size=self.batch_size,
                      cd_k=self.cd_k,
                      rbm_lr=self.rbm_lr,
                      dropout=self.dropout)
            dbn.build_model()
            self.dbn_list.append(dbn)
        # 构建dbn
        # 构建权值列表（dbn结构）
        # 权值 变量（初始化）
        ovon_b = tf.Variable(tf.constant(0.1, shape=[self.dbn_struct[-1]]),name='b_ovon')
        out_parameter_list=list()
        self.parameter_list = list()
        for dbn in self.dbn_list:
            ovon_W = tf.Variable(tf.truncated_normal(shape=[self.dbn_struct[-1], self.dbn_struct[-1]], stddev=0.1), name='W_ovon')
            out_parameter_list.append(ovon_W)
            self.parameter_list.append(dbn.parameter_list)
        self.parameter_list.append([out_parameter_list,ovon_b])
        # 损失函数
        self.pred=self.transform(self.input_data)
        _loss=Loss(label_data=self.label_data,
                 pred=self.pred,
                 output_act_func=self.output_act_func)
        self.loss=_loss.get_loss_func(self.loss_func)
        _optimization=Optimization(r=self.dbn_lr,
                                   momentum=self.momentum)
        self.train_batch_bp=_optimization.trainer(algorithm=self.bp_algorithm).minimize(self.loss, var_list=self.parameter_list)
        # 正确率
        _ac=Accuracy(label_data=self.label_data,
                 pred=self.pred)
        self.accuracy=_ac.accuracy()
        
    def train_model(self,X,train_list,sess):
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        # 训练子网络
        for i,dbn in enumerate(self.dbn_list):
            print("[Start training net-{}...]".format(i+1))
            dbn.train_model(train_X=train_list[i][0],train_Y=train_list[i][1],sess=sess)
        # 训练总网络
        print("[Start training output layer...]")            
        _data=Batch(images=X,
                    labels=train_list[0][1],
                    batch_size=self.batch_size)
        n=X.shape[0]
        # 迭代次数
        for i in range(self.dbn_epochs):
            for _ in range(int(n/self.batch_size)): 
                batch_x, batch_y= _data.next_batch()
                loss,_=sess.run([self.loss,self.train_batch_bp],feed_dict={self.input_data: batch_x,self.label_data: batch_y})
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_model(self,test_data,sess):
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy,self.pred],feed_dict={self.input_data: test_data[0],self.label_data: test_data[1]})
            print('[Accuracy]: %f' % acc)
            return pred_y
        else:
            loss,pred_y=sess.run([self.loss,self.pred],feed_dict={self.input_data: test_data[0],self.label_data: test_data[1]})
            print('[MSE]: %f' % loss)
            return pred_y
    
    def transform(self,X):
        dbn_output=list()
        for i,ovon_parameter in enumerate(self.parameter_list):
            if i<len(self.parameter_list)-1: # 对DBN循环
                next_data = X[:,i*self.dynamic:(i+1)*self.dynamic]
                next_data = self.dbn_list[i].transform(next_data)
                dbn_output.append(next_data)
            else:
                W_list=ovon_parameter[0]
                b=ovon_parameter[1]
                z=b
                for j,W in enumerate(W_list):
                    z=tf.add(tf.matmul(dbn_output[j], W), z)
                out_data=self.func_o(z)
        return out_data
    
    def gauss_func(self,x):
        return 1-tf.exp(-tf.square(x))
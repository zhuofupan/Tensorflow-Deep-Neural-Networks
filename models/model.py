# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../base")
from base_func import Batch,Loss,Accuracy,Optimization

class Model(object):
    def __init__(self,name):
        self.name = name
        self.momentum = 0.5
        self.output_act_func='softmax'
        self.loss_func='mse'
        self.bp_algorithm = 'sgd'
        self.best_acc = 0
        self.pt_model = None
        self.decay_lr = False
        self.loss = None
        self.accuracy = None
        self.train_batch = None
        self.merge = None
    
    def build_train_step(self):
        # 损失
        if self.loss is None:
            _loss=Loss(label_data=self.label_data,
                   pred=self.pred,
                   output_act_func=self.output_act_func)
            self.loss = _loss.get_loss_func(self.loss_func) # + 0.5*tf.matrix_determinant(tf.matmul(self.out_W,tf.transpose(self.out_W)))
        # 正确率
        if self.accuracy is None:
            
            _ac=Accuracy(label_data=self.label_data,
                     pred=self.pred)
            self.accuracy=_ac.accuracy()
            
        # 构建训练步
        if self.train_batch is None:
            if self.bp_algorithm=='adam' or self.bp_algorithm=='rmsp': 
                self.global_step =  None
                self.r = self.lr
            else: 
                self.global_step =  tf.Variable(0, trainable=False) # minimize 中会对 global_step 自加 1
                self.r = tf.train.exponential_decay(learning_rate=self.lr, global_step=self.global_step, decay_steps=100, decay_rate=0.96, staircase=True)
    
            self._optimization=Optimization(r=self.r,momentum=self.momentum)
            self.train_batch=self._optimization.trainer(algorithm=self.bp_algorithm).minimize(self.loss,global_step=self.global_step)
        
    def train_model(self,train_X,train_Y=None,val_X=None,val_Y=None,sess=None,summ=None,load_saver=''):
        pt_save_path='../saver/'+self.name+'/pre-train'
        ft_save_path='../saver/'+self.name+'/fine-tune'
        if not os.path.exists(pt_save_path): os.makedirs(pt_save_path)
        if not os.path.exists(ft_save_path): os.makedirs(ft_save_path)
        saver = tf.train.Saver()
        if load_saver=='f':
            # 加载训练好的模型
            print("Load Fine-tuned model...")
            saver.restore(sess,ft_save_path+'/fine-tune.ckpt')
            test_acc=self.validation_model(val_X,val_Y,sess)
            return print('>>> Test accuracy = {:.4}'.format(test_acc))
        elif load_saver=='p':
            # 加载预训练的模型
            print("Load Pre-trained model...")
            saver.restore(sess,pt_save_path+'/pre-train.ckpt')
        elif self.pt_model is not None:
            # 开始预训练
            print("Start Pre-training...")
            self.pt_model.train_model(train_X=train_X,sess=sess,summ=summ)
            print("Save Pre-trained model...")
            saver.save(sess,pt_save_path+'/pre-train.ckpt')
        # 开始微调
        print("Start Fine-tuning...")
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        
        b = int(train_X.shape[0]/self.batch_size)
        self.record_array=np.zeros((self.epochs,3))
        # 迭代次数
        for i in range(self.epochs):
            sum_loss=0; sum_acc=0
            for j in range(b):
                batch_x, batch_y= _data.next_batch()
                loss,acc,_=sess.run([self.loss,self.accuracy,self.train_batch],feed_dict={
                        self.input_data: batch_x,
                        self.label_data: batch_y,
                        self.keep_prob: 1-self.dropout})
                sum_loss = sum_loss + loss; sum_acc= sum_acc +acc
                
            #**************** 写入 ******************
            summary = sess.run(self.merge,feed_dict={self.input_data: batch_x,self.label_data: batch_y,self.keep_prob: 1-self.dropout})
            summ.train_writer.add_summary(summary, i)
            #****************************************
            loss = sum_loss/b; acc = sum_acc/b
            print('>>> epoch = {} , loss = {:.4} , accuracy = {:.4}'.format(i+1,loss,acc))
            self.record_array[i][0]=loss
            self.record_array[i][1]=acc
            if val_X is not None:
                val_acc=self.validation_model(val_X,val_Y,sess)
                print('    >>> validation accuracy = {:.4}'.format(val_acc))
                self.record_array[i][2]=val_acc
                           
        print("Save model...")
        saver.save(sess,ft_save_path+'/fine-tune.ckpt')
    
    def unsupervised_train_model(self,train_X,sess,summ):
        _data=Batch(images=train_X,
                    labels=None,
                    batch_size=self.batch_size)
        
        b = int(train_X.shape[0]/self.batch_size)
        # 迭代次数
        for i in range(self.epochs):
            sum_loss=0
            for j in range(b):
                if self.decay_lr:
                    self.lr = self.lr * 0.94
                batch_x = _data.next_batch()
                loss,_=sess.run([self.loss,self.train_batch],feed_dict={
                        self.input_data: batch_x,
                        self.label_data: batch_x})
                sum_loss = sum_loss + loss
    
            #**************** 写入 ******************
            summary = sess.run(self.merge,feed_dict={self.input_data: batch_x,self.label_data: batch_x})
            summ.train_writer.add_summary(summary, i)
            #****************************************
            loss = sum_loss/b
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_model(self,test_X,test_Y,sess):
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy,self.pred],feed_dict={
                    self.input_data: test_X,
                    self.label_data: test_Y,
                    self.keep_prob: 1.0})
            print('[Accuracy]: %f' % acc)
            return acc,pred_y
        else:
            mse,pred_y=sess.run([self.loss,self.pred],feed_dict={
                    self.input_data: test_X,
                    self.label_data: test_Y,
                    self.keep_prob: 1.0})
            print('[MSE]: %f' % mse)
            return mse,pred_y
        
    def validation_model(self,val_X,val_Y,sess):
        if type(val_X)==list: # TE 数据
            n_class = len(val_X)
            acc=np.zeros(n_class)
            pred_list=list()
            for i in range(n_class):
                if i==3 or i==9 or i==15: continue
                acc[i],pred=sess.run([self.accuracy,self.pred],feed_dict={
                        self.input_data: val_X[i],
                        self.label_data: val_Y[i],
                        self.keep_prob: 1.0})
                pred_list.append(pred)
            average_acc = np.sum(acc)/19
            
            if average_acc > self.best_acc:
                self.best_acc = average_acc
                self.best_acc_array = acc
                
                label_cnt=np.zeros((n_class,19))
                for i,pred in enumerate(pred_list):
                    label=np.argmax(pred,axis=1)
                    n_sample = label.shape[0]
                    for j in range(n_sample):
                        # 第 i 号分类 被 分到了 第 label[j] 号分类
                        label_cnt[label[j]][i]=label_cnt[label[j]][i]+1/n_sample
                self.label_distribution = label_cnt
                
            return average_acc
        else: # 手写识别
            n_class = val_Y.shape[1]
            
            acc,pred=sess.run([self.accuracy,self.pred],feed_dict={
                    self.input_data: val_X,
                    self.label_data: val_Y,
                    self.keep_prob: 1.0})
            
            if acc > self.best_acc:
                self.best_acc = acc
                pre_lab=np.argmax(pred,axis=1)
                real_lab=np.argmax(val_Y,axis=1)
                n_sample = pre_lab.shape[0]
                
                label_cnt=np.zeros((n_class,n_class))
                for i in range(n_sample):
                    # 第 real_lab[i] 号分类 被 分到了 第 pre_lab[i] 号分类
                    label_cnt[pre_lab[i]][real_lab[i]]=label_cnt[pre_lab[i]][real_lab[i]]+1
                sum_label = np.sum(label_cnt,axis=0)
                label_cnt = label_cnt /sum_label
                self.label_distribution = label_cnt
                
            return acc

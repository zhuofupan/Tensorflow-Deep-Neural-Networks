# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os

import sys
sys.path.append("../base")
from base_func import Batch,Loss,Accuracy,Optimization,plot_para_pic
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self,name):
        """
        user control
        """
        self.tbd = False
        self.sav = False
        self.show_pic =False
        self.plot_para=True
        
        # name
        self.name = name
        # record best acc
        self.ave_acc = 0
        self.acc_list = []
        
        # for unsupervised training
        self.momentum = 0.5
        self.output_act_func='softmax'
        self.loss_func='mse'
        self.bp_algorithm = 'rmsp'
        # for build train step
        self.pt_model = None
        self.decay_lr = False
        self.loss = None
        self.accuracy = None
        self.train_batch = None
        # for summary (tensorboard)
        self.merge = None
        # for plot
        self.pt_img =None
        # 用于预测
        self.pred_Y=None
        # 用于分类
        self.train_curve=None
        self.label_fig=None
        self.label_tag=None
    
    def build_train_step(self):
        # 损失
        if self.loss is None:
            _loss=Loss(label_data=self.label_data,
                       pred=self.pred,
                       logist=self.logist,
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
                self.r = tf.train.exponential_decay(learning_rate=self.lr, 
                                                    global_step=self.global_step, 
                                                    decay_steps=100, 
                                                    decay_rate=0.96, 
                                                    staircase=True)
#                self.global_step =  None
#                self.r = self.lr
    
            self._optimization=Optimization(r=self.r,momentum=self.momentum)
            self.train_batch=self._optimization.trainer(
                    algorithm=self.bp_algorithm).minimize(self.loss,global_step=self.global_step)
        
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
            self.pt_model.train_model(train_X=train_X,train_Y=train_Y,sess=sess,summ=summ)
            if self.sav:
                print("Save Pre-trained model...")
                saver.save(sess,pt_save_path+'/pre-train.ckpt')
            if self.plot_para:
                self.pt_img = sess.run(self.pt_model.parameter_list)
        # 开始微调
        print("Start Fine-tuning...")
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        
        b = int(train_X.shape[0]/self.batch_size)
        self.train_curve=np.zeros((self.epochs,3))
        self.label_tag=val_Y
        
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
            if self.tbd:
                summary = sess.run(self.merge,feed_dict={self.input_data: batch_x,self.label_data: batch_y,self.keep_prob: 1-self.dropout})
                summ.train_writer.add_summary(summary, i)
            #****************************************
            loss = sum_loss/b; acc = sum_acc/b
            
            self.train_curve[i][0]=loss
            if self.use_for=='classification':
                self.train_curve[i][1]=acc
                print('>>> epoch = {} , loss = {:.4} , accuracy = {:.4}'.format(i+1,loss,acc))
                if val_X is not None:
                    val_acc=self.validation_classification_model(val_X,val_Y,sess)
                    print('    >>> test accuracy = {:.4}'.format(val_acc))
                    self.train_curve[i][2]=val_acc
            else:
                print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
                    
        if self.use_for=='prediction':
            mse,self.pred_Y = self.test_model(val_X,val_Y,sess)
            self.test_Y = val_Y
            self.mse=mse
        
        if self.sav:                   
            print("Save model...")
            saver.save(sess,ft_save_path+'/fine-tune.ckpt')
        if self.plot_para:
            self.img = sess.run(self.parameter_list)
            plot_para_pic(self.pt_img,self.img,name=self.name)
    
    def unsupervised_train_model(self,train_X,train_Y,sess,summ):
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
            if self.tbd:
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
            return acc,pred_y
        else:
            mse,pred_y=sess.run([self.loss,self.pred],feed_dict={
                    self.input_data: test_X,
                    self.label_data: test_Y,
                    self.keep_prob: 1.0})
            return mse,pred_y
        
    def validation_classification_model(self,val_X,val_Y,sess):
        n_class = val_Y.shape[1]
        
        acc,pred=self.test_model(val_X,val_Y,sess)
        
        if acc > self.ave_acc:
            self.ave_acc = acc
            pre_lab=np.argmax(pred,axis=1)
            real_lab=np.argmax(val_Y,axis=1)
            self.label_fig=pre_lab
            n_sample = pre_lab.shape[0]
            
            label_cnt=np.zeros((n_class,n_class))
            for i in range(n_sample):
                # 第 real_lab[i] 号分类 被 分到了 第 pre_lab[i] 号分类
                label_cnt[pre_lab[i]][real_lab[i]]=label_cnt[pre_lab[i]][real_lab[i]]+1
            sum_label = np.sum(label_cnt,axis=0)
            label_cnt = label_cnt /sum_label
            self.label_distribution = label_cnt
            self.acc_list = np.diag(label_cnt)
        return acc
        
    def show_result(self,figname):
        if self.use_for=='classification':
            for i in range(len(self.acc_list)):
                print(">>>Test fault {}:".format(i))
                print('[Accuracy]: %f' % self.acc_list[i])
            print('[Average Accuracy]: %f' % self.ave_acc)
            self.plot_curve(figname) # 显示训练曲线
            self.plot_label_distribution() # 显示预测分布
            return self.label_distribution
        else:
            print('[MSE]: %f' % self.mse)
            self.plot_curve(figname) # 显示预测曲线
    
    def plot_curve(self,figname):
            fig = plt.figure(figsize=[32,18])
            plt.style.use('classic')
            if self.use_for=='classification':
                n = self.train_curve.shape[0]
                x = range(1,n+1)
                ax1 = fig.add_subplot(111)
                ax1.plot(x, self.train_curve[:,0],color='r',label='loss')
                ax1.set_ylabel('$Loss$')
                ax1.set_title("Training Curve")
                ax1.set_xlabel('$Epochs$')
                ax1.legend(loc='upper left')
                
                ax2 = ax1.twinx()  # this is the important function
                ax2.plot(x, self.train_curve[:,1],color='g',label='trian_acc')
                ax2.plot(x, self.train_curve[:,2],color='b',label='test_acc')
                ax2.set_ylabel('$Accuracy$')
                ax2.legend(loc='upper right')
            else:
                n = self.pred_Y.shape[0]
                x = range(1,n+1)
                ax1 = fig.add_subplot(111)
                ax1.plot(x, self.test_Y,color='r',label='test_Y')
                ax1.plot(x, self.pred_Y,color='g',label='pred_Y')
                ax1.set_title("Prediction Curve")
                ax1.set_xlabel('$point$')
                ax1.set_ylabel('$y$')
                ax1.legend(loc='upper right')
            
            if not os.path.exists('img'): os.makedirs('img')
            plt.savefig('img/'+figname+'.png',bbox_inches='tight')
            if self.show_pic: plt.show()
            plt.close(fig)
        
    def save_result(self):
        self.acc_list=list(self.acc_list)
        self.acc_list.append(self.ave_acc)
        np.savetxt("../saver/acc_list.csv", self.acc_list, fmt='%.4f',delimiter=",")
        np.savetxt("../saver/label_distribution.csv", self.label_distribution, fmt='%.4f',delimiter=",")
        np.savetxt("../saver/loss_and_acc.csv", self.train_curve, fmt='%.4f',delimiter=",")
    
    def plot_label_distribution(self):
        import warnings
        import matplotlib.cbook
        warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
        real_label = None
        pred_label = self.label_fig
        c = self.label_tag.shape[1] # 类数
        real_label = np.argmax(self.label_tag,axis=1)
        n = pred_label.shape[0] # 预测样本总数
        
        x = np.asarray(range(1,n+1))
        real_label = real_label.reshape(-1,)
        pred_label = pred_label.reshape(-1,)

        fig = plt.figure(figsize=[32,18])
        plt.style.use('ggplot')
        
        plt.yticks(range(c))
        
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, real_label,alpha=0.75,color='none', edgecolor='red', s=20,label='test_label')
        ax1.scatter(x, pred_label,alpha=0.75,color='none', edgecolor='blue', s=20,label='pred_label')
        ax1.set_title("Label Distribution")
        ax1.set_xlabel('$point$')
        ax1.set_ylabel('$label$')
        ax1.legend(loc='upper right')
        
        if not os.path.exists('img'): os.makedirs('img')
        plt.savefig('img/label_distibution.png',bbox_inches='tight')
        if self.show_pic: plt.show()
        plt.close(fig)
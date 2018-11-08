# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import sys
sys.path.append("../base")
from base_func import Batch,Loss,Accuracy,Optimization,plot_para_pic,tSNE_2d

class Model(object):
    def __init__(self,name):
        """
            ↓ user control ↓
        """
        self.show_pic =True       # show curve in 'Console'
        self.tbd = False          # tensorboard
        self.save_model = False   # save model
        self.plot_para=False      # plot W pic
        self.save_weight = False  # save W matrix
        self.do_tSNE = True       # t-SNE
        """
            ↑ user control ↑
        """
        # name
        self.name = name
        # record best acc
        self.ave_acc = 0          # average acc
        self.best_acc = None      # acc list
        
        # for pre-training
        self.momentum = 0.5
        self.output_act_func='softmax'
        self.loss_func='mse'
        self.bp_algorithm = 'rmsp'
        self.use_label = False    # supervised pre-training
        self.pre_exp_time = None  # pre-training expend time
        self.deep_feature = None  
        # for fine-tuning
        self.h_act_p = 0
        self.recon_data = None
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
        self.title = False
        # for 'prediction'
        self.pred_Y=None
        self.mse = np.inf
        # for 'classification'
        self.loss_and_acc=None     # loss, train_acc, test_acc, spend_time
        self.test_Y=None           # real label
        self.real_class = None
        self.pred_class = None
        

    #########################
    #        Build          #
    #########################
    
    
    def build_train_step(self):
        # 预训练/微调
        if self.recon_data is not None:
            label_data = self.recon_data
        else: 
            label_data = self.label_data
        # 损失
        if self.loss is None:
            _loss=Loss(label=label_data, 
                       logits=self.logits,
                       out_func_name=self.output_act_func,
                       loss_name=self.loss_func)
            self.loss = _loss.get_loss_func() # + 0.5*tf.matrix_determinant(tf.matmul(self.out_W,tf.transpose(self.out_W)))
        # 正确率
        if self.accuracy is None:
            
            _ac=Accuracy(label_data=label_data,
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
    
            self._optimization=Optimization(r=self.r,momentum=self.momentum)
            self.train_batch=self._optimization.trainer(
                    algorithm=self.bp_algorithm).minimize(self.loss,global_step=self.global_step)


    #########################
    #        Train          #
    #########################
    

    def train_model(self,train_X,train_Y=None,test_X=None,test_Y=None,sess=None,summ=None,load_saver=''):
        pt_save_path='../saver/'+self.name+'/pre-train'
        ft_save_path='../saver/'+self.name+'/fine-tune'
        if not os.path.exists(pt_save_path): os.makedirs(pt_save_path)
        if not os.path.exists(ft_save_path): os.makedirs(ft_save_path)
        saver = tf.train.Saver()
        W_csv_pt = None
        if load_saver=='f':
            # 加载训练好的模型 --- < fine-tuned >
            print("Load Fine-tuned model...")
            saver.restore(sess,ft_save_path+'/fine-tune.ckpt')
            # >>> Test Acc
            test_acc=self.validation_model(test_X,test_Y,sess) 
            return print('>>> Test accuracy = {:.4}'.format(test_acc))
        elif load_saver=='p':
            # 加载预训练的模型 --- < pre-trained >
            print("Load Pre-trained model...")
            saver.restore(sess,pt_save_path+'/pre-train.ckpt')
        elif self.pt_model is not None:
            
            #####################################################################
            #     开始逐层预训练 -------- < start pre-traning layer by layer>     #
            #####################################################################
            
            print("Start Pre-training...")
            pre_time_start=time.time()
            # >>> Pre-traning -> unsupervised_train_model
            self.deep_feature = self.pt_model.train_model(train_X=train_X,train_Y=train_Y,sess=sess,summ=summ)
            pre_time_end=time.time()
            self.pre_exp_time = pre_time_end-pre_time_start
            print('>>> Pre-training expend time = {:.4}'.format(self.pre_exp_time))
            
            if self.save_weight:
                W_csv_pt = self.save_modele_weight_csv('pt',sess)
            if self.save_model:
                print("Save Pre-trained model...")
                saver.save(sess,pt_save_path+'/pre-train.ckpt')
            if self.use_for=='classification' and self.do_tSNE:
                tSNE_2d(self.deep_feature,train_Y,'train')
                if test_Y is not None:
                    test_deep_feature = sess.run(self.pt_model.transform(test_X))
                    tSNE_2d(test_deep_feature,test_Y,'test')
        
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        
        b = int(train_X.shape[0]/self.batch_size)
        
        # 统计样本总数
        if self.use_for=='classification' and test_Y is not None:
            real_class = []
            if type(test_Y)==list:
                self.sample_cnt=np.zeros([test_Y[0].shape[1]]) # 各类样本总数
                self.sample_total = 0                          # 所有样本总数
                for i,Y in enumerate(test_Y):
                    if i==0:
                        real_class = np.argmax(Y,axis=1)
                    else:
                        real_class = np.concatenate((real_class,np.argmax(Y,axis=1)),axis=0)
                    self.sample_cnt[i] = test_Y[i].shape[0]
                    self.sample_total = self.sample_total + self.sample_cnt[i]
            self.real_class = real_class
                
        self.loss_and_acc=np.zeros((self.epochs,4))
        self.test_Y=test_Y  

        #######################################################
        #     开始微调 -------------- < start fine-tuning >    #
        #######################################################
        
        print("Start Fine-tuning...")
        # 迭代次数
        time_start=time.time()
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
            
            self.loss_and_acc[i][0]=loss              # <0> 损失loss
            time_end=time.time()
            time_delta = time_end-time_start
            self.loss_and_acc[i][3]=time_delta        # <3> 耗时time
            
            # >>> for 'classification'
            if self.use_for=='classification': 
                self.loss_and_acc[i][1]=acc           # <1> 训练acc
                string = '\r'+'>>> epoch = {}/{} , loss = {:.4} , train accuracy = {:.4}% , expend time = {:.4}'.format(i+1,self.epochs,loss,acc*100,time_delta)
                sys.stdout.write(string)
                sys.stdout.flush()
                
                ###########################################################
                #     开始测试    <classification>  with: test_X, test_Y   #
                ###########################################################
                
                if test_Y is not None:
                    acc=self.test_average_accuracy(test_X,test_Y,sess)
                    string = string + '\n' +  '    >>> test accuracy = {:.4}%'.format(acc*100)
                    sys.stdout.write(string)
                    sys.stdout.flush()
                    self.loss_and_acc[i][2]=acc       # <2> 测试acc
            # >>> for 'prediction'
            else:                              
                string = '\r'+'>>> epoch = {}/{} , loss = {:.4}'.format(i+1,self.epochs,loss)
                sys.stdout.write(string)
                sys.stdout.flush()
                
                ###########################################################
                #     开始测试    <prediction>  with: test_X, test_Y       #
                ###########################################################
                
                if test_Y is not None:
                    mse,pred_Y = self.test_model(test_X,test_Y,sess)
                    string = string + '\n' +  '    >>> test mse = {:.4}%'.format(mse)
                    sys.stdout.write(string)
                    sys.stdout.flush()
                    self.loss_and_acc[i][2]=mse       # <2> 测试acc
                    if mse < self.mse:
                        self.mse = mse
                        self.pred_Y = pred_Y

        print('')
        
        #################################################################
        #     开始测试    <classification, prediction>  with: test_X     #
        #################################################################
        
        if test_X is not None and test_Y is None:
            if self.use_for=='classification':
                _,pred = self.test_model(test_X,test_Y,sess)
                self.pred_class=np.argmax(pred,axis=1)
            else:
                _,self.pred_Y = self.test_model(test_X,test_Y,sess)
        
        if self.save_weight:
            W_csv_ft = self.save_modele_weight_csv('ft',sess)
            
        if self.plot_para:  
            plot_para_pic(W_csv_pt,W_csv_ft,name=self.name)
            
        if self.save_model:                   
            print("Save model...")
            saver.save(sess,ft_save_path+'/fine-tune.ckpt')
        
        np.savetxt("../saver/loss_and_acc.csv", self.loss_and_acc, fmt='%.4f',delimiter=",")
    
    def unsupervised_train_model(self,train_X,train_Y,sess,summ):
        if self.use_label: labels = train_Y
        else: labels = None
        _data=Batch(images=train_X,
                    labels=labels,
                    batch_size=self.batch_size)
        
        b = int(train_X.shape[0]/self.batch_size)
        
        ########################################################
        #     开始训练 -------- < start traning for rbm/ae>     #
        ########################################################
        
        # 迭代次数
        for i in range(self.epochs):
            sum_loss=0
            if self.decay_lr:
                self.lr = self.lr * 0.94
            for j in range(b):
                # 有监督
                if self.use_label:  
                    batch_x,batch_y = _data.next_batch()
                    loss,_=sess.run([self.loss,self.train_batch],feed_dict={
                        self.input_data: batch_x,
                        self.recon_data: batch_x,
                        self.label_data: batch_y})
                # 无监督
                else:   
                    batch_x = _data.next_batch()
                    loss,_=sess.run([self.loss,self.train_batch],feed_dict={
                            self.input_data: batch_x,
                            self.recon_data: batch_x})
                sum_loss = sum_loss + loss
    
            #**************** 写入 ******************
            if self.tbd:
                summary = sess.run(self.merge,feed_dict={self.input_data: batch_x,self.label_data: batch_x})
                summ.train_writer.add_summary(summary, i)
            #****************************************
            loss = sum_loss/b
            string = '\r'+'>>> epoch = {}/{} , loss = {:.4}'.format(i+1,self.epochs,loss)
            sys.stdout.write(string)
            sys.stdout.flush()
            
        print('')
    
    
    #########################
    #        Judge          #
    #########################
    
    
    def test_model(self,test_X,test_Y,sess):
        pred_y=sess.run(self.pred,feed_dict={
                        self.input_data: test_X,
                        self.keep_prob: 1.0})
        if test_Y is None:
            return None,pred_y
        if self.use_for=='classification':
            acc=sess.run(self.accuracy,feed_dict={
                         self.input_data: test_X,
                         self.label_data: test_Y,
                         self.keep_prob: 1.0})
            return acc,pred_y
        else:
            mse=sess.run(self.loss,feed_dict={
                        self.input_data: test_X,
                        self.label_data: test_Y,
                        self.keep_prob: 1.0})
            return mse,pred_y
    
    # for 'array' test data
    def test_average_accuracy(self,test_X,test_Y,sess):
        # 图片分类任务
        n_class = test_Y.shape[1]
        
        acc,pred=self.test_model(test_X,test_Y,sess)
        
        if acc > self.ave_acc:
            self.ave_acc = acc
            pred_class=np.argmax(pred,axis=1)
            real_class=self.real_class
            self.pred_class=pred_class
            n_sample = pred_class.shape[0]
            
            pred_cnt=np.zeros((n_class,n_class))
            for i in range(n_sample):
                # 第 real_class[i] 号分类 被 分到了 第 pred_class[i] 号分类
                pred_cnt[pred_class[i]][real_class[i]]=pred_cnt[pred_class[i]][real_class[i]]+1
            sum_label = np.sum(pred_cnt,axis=0)
            pred_per = pred_cnt /sum_label
            self.label_distribution = pred_per
            self.best_acc = np.diag(pred_per) # array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
                                              # array是一个2维矩阵时，结果输出矩阵的对角线元素 <这里是这种情况>
        return acc
    
    ##########################
    #        Result          #
    ##########################
        
    def show_and_save_result(self,figname):
         
        if self.test_Y is not None:
            
            if self.use_for=='classification':
                for i in range(len(self.best_acc)):
                    print(">>>Test Class {}:".format(i+1))
                    print('[Accuracy]: {:.4}%'.format(self.best_acc[i]*100))
                print('[Average accuracy]: {:.4}%'.format(self.ave_acc*100))
                self.plot_label_distribution()   # 显示预测分布
                
            self.plot_curve(figname)             # 显示训练/预测曲线
            print("Save csv...")
            
            if self.use_for=='classification':
                np.savetxt("../saver/best_acc.csv", self.best_acc, fmt='%.4f',delimiter=",")
                np.savetxt("../saver/label_distribution.csv", self.label_distribution, fmt='%.4f',delimiter=",")
                np.savetxt("../saver/real_class.csv", self.real_class, fmt='%.4f',delimiter=",")
                np.savetxt("../saver/pred_class.csv", self.pred_class, fmt='%.4f',delimiter=",")          

    #######################
    #        Weight       #
    #######################
    
    def save_modele_weight_csv(self,stage,sess):  # save W csv
        print("Save weight...")
        if not os.path.exists('../saver/weight'): os.makedirs('../saver/weight')
        if stage == 'pt':
            para_list = self.pt_model.parameter_list
        else:
            para_list = self.parameter_list
            
        W_list=list()
        for i in range(len(para_list)):
            W = para_list[i][0]
            np_W = sess.run(W)
            np.savetxt("../saver/weight/["+stage+"]W_"+str(i+1)+".csv", np_W, fmt='%.4f',delimiter=",")
            W_list.append(np_W)
            
        return W_list

    ########################
    #        Plot          #
    ########################
    
    def plot_curve(self,figname):
        import matplotlib.pyplot as plt
        
        plt.style.use('classic')
        fig = plt.figure(figsize=[32,18])

        if self.use_for=='classification':
            print("Plot loss and acc curve...")
            n = self.loss_and_acc.shape[0]
            x = range(1,n+1)
            ax1 = fig.add_subplot(111)
            ax1.plot(x, self.loss_and_acc[:,0],color='r',marker='o',markersize=10,linestyle='-.',linewidth=4,label='loss')
            ax1.set_ylabel('$Loss$',fontsize=36)
            if self.title:
                ax1.set_title("Training loss and test accuracy")
            ax1.set_xlabel('$Epochs$',fontsize=36)
            ax1.legend(loc='upper left',fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            
            ax2 = ax1.twinx()  # this is the important function
            ax2.plot(x, self.loss_and_acc[:,1],color='c',marker='D',markersize=10,linestyle='-',linewidth=4,label='train_acc')
            if self.test_Y is not None:
                ax2.plot(x, self.loss_and_acc[:,2],color='m',marker='D',markersize=10,linestyle='-',linewidth=4,label='test_acc')
            ax2.set_ylabel('$Accuracy$',fontsize=36)
            ax2.legend(loc='upper right',fontsize=24)
            plt.yticks(fontsize=20)
        else:
            print("Plot prediction curve...")
            n = self.pred_Y.shape[0]
            x = range(1,n+1)
            ax1 = fig.add_subplot(111)
            ax1.plot(x, self.test_Y,color='r',marker='D',markersize=10,linestyle='-',linewidth=4,label='test_Y')
            ax1.plot(x, self.pred_Y,color='g',marker='D',markersize=10,linestyle='-',linewidth=4,label='pred_Y')
            if self.title:
                ax1.set_title("prediction curve")
            ax1.set_xlabel('$sample$',fontsize=36)
            ax1.set_ylabel('$y$',fontsize=36)
            ax1.legend(loc='upper right')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        
        if not os.path.exists('../saver/img'): os.makedirs('../saver/img')
        plt.savefig('../saver/img/'+figname+'.png',bbox_inches='tight')
        if self.show_pic: plt.show()
        plt.close(fig)
    
    def plot_label_distribution(self):
        import warnings
        import matplotlib.cbook
        import matplotlib.pyplot as plt
        warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
        
        print("Plot label distribution...")
        
        real_class = self.real_class
        pred_class = self.pred_class
        
        n = pred_class.shape[0] # 预测样本总数
        x = np.asarray(range(1,n+1))
        real_class = real_class.reshape(-1,)
        pred_class = pred_class.reshape(-1,)
        
        fig = plt.figure(figsize=[32,18])
        plt.style.use('ggplot')
        
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, real_class,alpha=0.75,color='none', edgecolor='red', s=20,label='test_class')
        ax1.scatter(x, pred_class,alpha=0.75,color='none', edgecolor='blue', s=20,label='pred_class')
        if self.title:
            ax1.set_title("Label Distribution",fontsize=36)
        ax1.set_xlabel('$point$',fontsize=36)
        ax1.set_ylabel('$label$',fontsize=36)
        ax1.legend(loc='upper left',fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        
        if not os.path.exists('../saver/img'): os.makedirs('../saver/img')
        plt.savefig('../saver/img/label_distibution.png',bbox_inches='tight')
        if self.show_pic: plt.show()
        plt.close(fig)
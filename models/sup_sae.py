# -*- coding: utf-8 -*-
import tensorflow as tf
from un_sae import unsupervised_sAE
import sys
sys.path.append("../base")
from base_func import Batch,Loss,Accuracy,Optimization,act_func,Summaries

class supervised_sAE(object):
    def __init__(self,
                 out_func='softmax',
                 en_func='sigmoid', # encoder：[sigmoid] 
                 use_for='classification',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [relu] with ‘mse’
                 ae_type='ae', # ae | dae | sae
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5,  # 惩罚因子权重（第二项损失的系数）
                 p=0.5, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 sup_ae_struct=[784, 100, 100, 10],
                 sup_ae_epochs=100,
                 ae_epochs=10,
                 batch_size=32,
                 ae_lr=1e-3,
                 dropout=1):
        self.out_func=out_func
        self.en_func=en_func
        self.use_for=use_for
        self.loss_func=loss_func
        self.ae_type = ae_type
        self.noise_type = noise_type
        self.beta = beta
        self.p = p
        self.sup_ae_struct = sup_ae_struct
        self.un_ae_struct = sup_ae_struct[:-1]
        self.sup_ae_epochs=sup_ae_epochs
        self.ae_epochs = ae_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        self.dropout = dropout
        self.hidden_act=act_func(self.en_func)
        self.output_act=act_func(self.out_func)
        
        self.build_model()
        
    #######################
    #    sup_sAE_model    #
    #######################
    
    def build_model(self): 
        """
        Pre-training
        """
        # 构建un_sae
        self.un_sae = unsupervised_sAE(
                en_func=self.en_func,
                loss_func=self.loss_func, # encoder：[sigmoid] || decoder：[sigmoid] with ‘cross_entropy’ | [relu] with ‘mse’
                ae_type=self.ae_type, # ae | dae | sae
                noise_type=self.noise_type, # Gaussian noise (gs) | Masking noise (mn)
                beta=self.beta,  # 惩罚因子权重（第二项损失的系数）
                p=self.p, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                un_ae_struct=self.un_ae_struct,
                ae_epochs=self.ae_epochs,
                batch_size=self.batch_size,
                ae_lr=self.ae_lr)
        
        """
        Fine-tuning
        """
        with tf.name_scope('sup_sAE'):
            # feed 变量
            self.input_data = tf.placeholder(tf.float32, [None, self.sup_ae_struct[0]],name='X') # N等于batch_size（训练）或_num_examples（测试）
            self.label_data = tf.placeholder(tf.float32, [None, self.sup_ae_struct[-1]],name='Y') # N等于batch_size（训练）或_num_examples（测试）
            # 权值 变量（初始化）
            self.out_W = tf.Variable(tf.truncated_normal(shape=[self.sup_ae_struct[-2], self.sup_ae_struct[-1]], stddev=0.1), name='W-out')
            self.out_b = tf.Variable(tf.constant(0.1, shape=[self.sup_ae_struct[-1]]),name='b-out')
            # 构建sup_sae
            # 构建权值列表（sup_sae结构）
            self.parameter_list = list()
            for ae in self.un_sae.ae_list:
                self.parameter_list.append(ae.W)
                self.parameter_list.append(ae.by)
            self.parameter_list.append(self.out_W)
            self.parameter_list.append(self.out_b)
            
            # 损失函数
            self.pred=self.transform(self.input_data)
            _loss=Loss(label_data=self.label_data,
                     pred=self.pred,
                     output_act_func=self.out_func)
            self.loss=_loss.get_loss_func(self.loss_func)
            _optimization=Optimization(r=self.ae_lr,
                                       momentum=0.5)
            self.train_batch_bp=_optimization.trainer(algorithm='sgd').minimize(self.loss, var_list=self.parameter_list)
            # 正确率
            _ac=Accuracy(label_data=self.label_data,
                     pred=self.pred)
            self.accuracy=_ac.accuracy()
            
            #****************** 记录 ******************
            for i in range(len(self.parameter_list)):
                if i%2==1:continue
                k=int(i/2+1)
                W=self.parameter_list[i]
                b=self.parameter_list[i+1]
                Summaries.scalars_histogram('_W'+str(k),W)
                Summaries.scalars_histogram('_b'+str(k),b)
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,tf.get_default_graph()._name_stack))
            #******************************************
        
    def train_model(self,train_X,train_Y,sess,summ):
        # 预训练
        print("Start Pre-training...")
        self.un_sae.train_model(train_X,sess=sess,summ=summ)
        # 微调
        print("Start Fine-tuning...")  
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        m=int(n/self.batch_size)
        mod=max(int(self.rbm_epochs*m/1000),1)
        # 迭代次数
        k=0
        for i in range(self.sup_ae_epochs):
            for _ in range(m): 
                k=k+1
                batch_x, batch_y= _data.next_batch()
                summary,loss,_=sess.run([self.merge,self.loss,self.train_batch_bp],
                                        feed_dict={self.input_data: batch_x,self.label_data: batch_y})
                
                #**************** 写入 ******************
                if k%mod==0: summ.train_writer.add_summary(summary, k)
                #****************************************
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def test_model(self,test_X,test_Y,sess):
        self.dropout=1.0
        if self.use_for=='classification':
            acc,pred_y=sess.run([self.accuracy,self.pred],
                                feed_dict={self.input_data: test_X,self.label_data: test_Y})
            
            print('[Accuracy]: %f' % acc)
            return pred_y
        else:
            loss,pred_y=sess.run([self.loss,self.pred],
                                 feed_dict={self.input_data: test_X,self.label_data: test_Y})
            
            print('[MSE]: %f' % loss)
            return pred_y
    
    def transform(self,data_x):
        # 得到网络输出值
        next_data = data_x # 这个next_data是tf变量
        for i in range(len(self.parameter_list)):
            if i%2==1:continue
            W=self.parameter_list[i]
            b=self.parameter_list[i+1]
            z = tf.add(tf.matmul(next_data, W), b)
            if i==len(self.parameter_list)-1:
                next_data=self.output_act(z)
            else:
                next_data=self.hidden_act(z)
            next_data = tf.nn.dropout(next_data, self.dropout)
        return next_data
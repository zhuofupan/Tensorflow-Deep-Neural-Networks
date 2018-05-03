# -*- coding: utf-8 -*-
import tensorflow as tf
from un_sae import unsupervised_sAE
import sys
sys.path.append("../base")
from model import Model
from base_func import act_func,Summaries

class supervised_sAE(Model):
    def __init__(self,
                 out_func='softmax',
                 en_func='affine', # encoder：[sigmoid] | [affine] 
                 use_for='classification',
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
                 struct=[784, 100, 100, 10],
                 lr=1e-4,
                 epochs=60,
                 batch_size=32,
                 dropout=0,
                 ae_type='dae', # ae | dae | sae
                 noise_type='mn', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.25,  # 惩罚因子权重（KL项 | 非噪声样本项）
                 p=0.5, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）      
                 ae_lr=1e-3,
                 ae_epochs=20):
        Model.__init__(self,'sup_sAE')
        self.out_func=out_func
        self.en_func=en_func
        self.use_for=use_for
        self.loss_func=loss_func
        self.struct = struct
        self.lr = lr
        self.epochs=epochs
        self.dropout = dropout
        
        self.un_ae_struct = struct[:-1]
        
        self.ae_type = ae_type
        self.noise_type = noise_type
        self.beta = beta
        self.p = p
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.batch_size = batch_size
        
        self.hidden_act=act_func(self.en_func)
        self.output_act=act_func(self.out_func)
        
        self.build_model()
        
    #######################
    #    sup_sAE_model    #
    #######################
    
    def build_model(self): 
        print("Start building model...")
        print('sup_sAE:')
        print(self.__dict__)
        """
        Pre-training
        """
        # 构建un_sae
        self.pt_model = unsupervised_sAE(
                en_func=self.en_func,
                loss_func=self.loss_func, # encoder：[sigmoid] || decoder：[sigmoid] with ‘cross_entropy’ | [relu] with ‘mse’
                ae_type=self.ae_type, # ae | dae | sae
                noise_type=self.noise_type, # Gaussian noise (gs) | Masking noise (mn)
                beta=self.beta,  # 惩罚因子权重（第二项损失的系数）
                p=self.p, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                un_ae_struct=self.un_ae_struct,
                ae_epochs=self.ae_epochs,
                batch_size=self.batch_size,
                lr=self.ae_lr)
        
        """
        Fine-tuning
        """
        with tf.name_scope('sup_sAE'):
            # feed 变量
            self.input_data = tf.placeholder(tf.float32, [None, self.struct[0]],name='X') # N等于batch_size（训练）或_num_examples（测试）
            self.label_data = tf.placeholder(tf.float32, [None, self.struct[-1]],name='Y') # N等于batch_size（训练）或_num_examples（测试）
            self.keep_prob = tf.placeholder(tf.float32) 
            # 权值 变量（初始化）
            self.out_W = tf.Variable(tf.truncated_normal(shape=[self.struct[-2], self.struct[-1]], stddev=0.1), name='W-out')
            self.out_b = tf.Variable(tf.constant(0.1, shape=[self.struct[-1]]),name='b-out')
            # 构建sup_sae
            # 构建权值列表（sup_sae结构）
            self.parameter_list = list()
            for pt in self.pt_model.pt_list:
                self.parameter_list.append(pt.W)
                self.parameter_list.append(pt.by)
            self.parameter_list.append(self.out_W)
            self.parameter_list.append(self.out_b)
            
            # 构建训练步
            self.pred=self.transform(self.input_data)
            self.build_train_step()
            
            #****************** 记录 ******************
            for i in range(len(self.parameter_list)):
                if i%2==1:continue
                k=int(i/2+1)
                Summaries.scalars_histogram('_W'+str(k),self.parameter_list[i])
                Summaries.scalars_histogram('_b'+str(k),self.parameter_list[i+1])
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            self.merge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'sup_sAE'))
            #******************************************
    
    def transform(self,data_x):
        # 得到网络输出值
        next_data = data_x # 这个next_data是tf变量
        for i in range(len(self.parameter_list)):
            if i%2==1:continue
            W=self.parameter_list[i]
            b=self.parameter_list[i+1]
            z = tf.add(tf.matmul(next_data, W), b)
            if i==len(self.parameter_list)-2:
                next_data=self.output_act(z)
            else:
                next_data=self.hidden_act(z)
            if self.dropout<1:
                next_data = tf.nn.dropout(next_data, self.dropout)
        return next_data
# -*- coding: utf-8 -*-
from ae import AE

class unsupervised_sAE(object):
    def __init__(self,
                 loss_func='mse', # decoder：[sigmoid] with ‘cross_entropy’ | [relu] with ‘mse’
                 ae_type='ae', # ae | dae | sae
                 act_type=['sigmoid','affine'],
                 noise_type='gs', # Gaussian noise (gs) | Masking noise (mn)
                 beta=0.5,  # 惩罚因子权重（第二项损失的系数）
                 p=0.5, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                 struct=[784, 100, 100],
                 ae_epochs=10,
                 batch_size=32,
                 ae_lr=1e-3):
        self.act_type=act_type
        self.loss_func=loss_func
        self.ae_type = ae_type
        self.noise_type = noise_type
        self.beta = beta
        self.p = p
        self.struct = struct[:-1]
        self.out_size = struct[-1]
        self.ae_epochs = ae_epochs
        self.batch_size = batch_size
        self.ae_lr = ae_lr
        
        self.build_model()
        
    ######################
    #    un_sAE_model    #
    ######################
    
    def build_model(self):
        # 构建rmbs
        self.pt_list = list()
        self.parameter_list=list()
        for i in range(len(self.struct) -1):
            print('Build AE-{}...'.format(i+1))
            n_x = self.struct[i]
            n_y = self.struct[i+1]
            if self.ae_type=='sae' and n_x>n_y: ae_type='ae'
            else: ae_type=self.ae_type
            name=ae_type+'-'+ str(i + 1)
            ae = AE(name=name,
                    act_type=self.act_type,
                    loss_func=self.loss_func, # encoder：[sigmoid] || decoder：[sigmoid] with ‘cross_entropy’ | [relu] with ‘mse’
                    ae_type=ae_type, # ae | dae | sae
                    noise_type=self.noise_type, # Gaussian noise (gs) | Masking noise (mn)
                    beta=self.beta,  # 惩罚因子权重（第二项损失的系数）
                    p=self.p, # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
                    struct=[n_x,n_y],
                    out_size = self.out_size,
                    ae_epochs=self.ae_epochs,
                    batch_size=self.batch_size,
                    lr=self.ae_lr)
            # print(ae.__dict__)
            self.pt_list.append(ae) # 加入list
            self.parameter_list.append([ae.W,ae.bh])
            
    def train_model(self,train_X,train_Y,sess,summ):
        # 返回最后一层特征<实值>
        X = train_X 
        for i,ae in enumerate(self.pt_list):
            print('>>> Training AE-{}:'.format(i+1))
            # 训练第i个AE（按batch）
            ae.unsupervised_train_model(X,train_Y,sess=sess,summ=summ)
            # 得到transform值（train_X）
            X = sess.run(ae.transform(X))
        return X
    
    def transform(self,X):
        # 返回最后一层特征<tf变量>
        for ae in self.pt_list:
            X = ae.transform(X)
        return X
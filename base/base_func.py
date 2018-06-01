# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

class Batch(object):
    def __init__(self,
                 images=None,
                 labels=None,
                 batch_size=None,
                 shuffle=True):
        self.images = images
        if labels is None:
            self.exit_y = False
        else:
            self.exit_y = True
            self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    def next_batch(self):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            if self.exit_y: self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + self.batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            if self.exit_y: labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if self.shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                if self.exit_y: self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            if self.exit_y:
                labels_new_part = self._labels[start:end]
                return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
            else:
                return np.concatenate((images_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += self.batch_size
            end = self._index_in_epoch
            if self.exit_y:
                return self._images[start:end], self._labels[start:end]
            else:
                return self._images[start:end]

def act_func(func_name):
    if func_name=='sigmoid':   # S(z) = 1/(1+exp(-z)) ∈ (0,1)
        return tf.nn.sigmoid
    elif func_name=='softmax': # s(z) = S(z)/∑S(z) ∈ (0,1)
        return tf.nn.softmax
    elif func_name=='relu':    # r(z) = max(0,z) ∈ (0,+inf)
        return tf.nn.relu
    elif func_name=='tanh':    # r(z) = max(0,z) ∈ (0,+inf)
        return tf.nn.tanh
    elif func_name=='elu':
        return tf.nn.elu
    elif func_name=='selu':
        return tf.nn.selu
    elif func_name=='gauss':   # g(z) = 1-exp(-z^2) ∈ (0,1)
        def gauss(z):
            return 1-tf.exp(-tf.square(z))
        return gauss
    elif func_name=='affine':
        def affine(z):
            return z
        return affine
    elif func_name=='tanh2':
        def tanh2(z):
            return (1-tf.exp(-tf.square(z)))/(1+tf.exp(-tf.square(z)))
        return tanh2
    elif func_name=='standardization':
        def standardization(z):
            mean,variance=tf.nn.moments(z,axes=0)
            return (z-mean)/variance
        return standardization

def np_func(func_name):
    if func_name=='standardization': # 0均值1方差化
        def standardization(z):
            mean = np.mean(z,axis=0)
            var = np.var(z,axis=0)
            return (z-mean)/var
        return standardization
    elif func_name=='l2_normalize': # 单位向量化
        def l2_normalize(z):
            return z / np.sqrt(max(np.sum(z**2,axis=0),1e-12))
        return l2_normalize

def BN(var,z,gamma,beta):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    # mean, variance = tf.nn.moments(var)
    z=(z-mean)/stddev
    z=gamma*z+beta
    return z

def init_rest_var(sess):
    uninit_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninit_vars.append(var)
    sess.run(tf.variables_initializer(uninit_vars))  
    
class Summaries(object):
    def __init__(self,
                 file_name,
                 sess):
        # 定义 FileWriter 用于记录 merge
        write_path = '../tensorboard/'+file_name
        if not os.path.exists(write_path): os.makedirs(write_path)
        self.train_writer = tf.summary.FileWriter(write_path, sess.graph)
   
    def scalars_histogram(name,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
          # 计算参数的均值，并使用tf.summary.scaler记录
          mean = tf.reduce_mean(var)
          #tf.summary.scalar('mean', mean)
          # 计算参数的标准差
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('mean', mean)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          # 用直方图记录参数的分布
          tf.summary.histogram('distribution', var)

class Loss(object):
    def __init__(self,
                 label_data,
                 pred,
                 logist,
                 output_act_func):
        self.label_data = label_data
        self.pred = pred
        self.logist = logist
        self.output_act_func = output_act_func  
        
    def get_loss_func(self,func_name):
        if func_name=='cross_entropy':
            if self.output_act_func=='softmax':
                return tf.losses.softmax_cross_entropy(self.label_data, self.logist)
            if self.output_act_func=='sigmoid':
                return tf.losses.sigmoid_cross_entropy(self.label_data, self.logist)
        if func_name=='mse':
            return tf.losses.mean_squared_error(self.label_data, self.pred)
        
class Accuracy(object):
    def __init__(self,
                 label_data,
                 pred):
        self.label_data = label_data
        self.pred = pred
        
    def accuracy(self):
        if self.label_data.shape[1]>1:
            pre_lables=tf.argmax(self.pred,axis=1)
            data_lables=tf.argmax(self.label_data,axis=1)
        else:
            pre_lables=tf.round(self.pred)
            data_lables=tf.round(self.label_data)
        return tf.reduce_mean(tf.cast(tf.equal(pre_lables,data_lables),tf.float32))
    
class Optimization(object):
    def __init__(self,r=1e-3,momentum=0.5,use_nesterov=False):
        self.r = r
        self.momentum = momentum
        self.use_nesterov=use_nesterov
    
    def trainer(self,algorithm='sgd'):
        if algorithm=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.r)
        elif algorithm == 'adag':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.r,
                                                  initial_accumulator_value=0.1)
        elif algorithm == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.r,
                                               beta1=0.9,
                                               beta2=0.999,
                                               epsilon=1e-8)
        elif algorithm == 'mmt':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.r,
                                                   momentum=self.momentum,
                                                   use_nesterov=self.use_nesterov)
        elif algorithm == 'rmsp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.r,
                                                  momentum=self.momentum)
        return optimizer

def plot_para_pic(pt_img,img,name):

    plt.style.use('classic')
    for i in range(len(img)):
        fsize = np.asarray(img[i][0].shape,dtype=np.float32)
        while fsize[0]>=9*4 or fsize[1]>=16*4: fsize=fsize*0.5
        while fsize[0]<9 or fsize[1]<16: fsize=fsize*2
        
        if pt_img is not None and i<len(img)-1:
            fig = plt.figure(figsize=[fsize[1]*2,fsize[0]])
            cnt=2
        else:
            fig = plt.figure(figsize=[fsize[1],fsize[0]])
            cnt=1
        
        for k in range(cnt):
            ax = fig.add_subplot(1,cnt,k+1)
            if k==0: 
                data = img[i][0]
                ax.set_title("Fine-tuned")
            else: 
                data = pt_img[i][0]
                ax.set_title("Pre-trained")
            """
            cmap 主题:
                ocean:海洋
                hot:火热
                gray:灰度
                spectral:光谱
                paired:配对
                jet:彩色
                rainbow:彩虹
            """
            
            im = ax.imshow(data,interpolation='nearest',cmap=plt.cm.rainbow,origin='lower')
            ax.set_xticks(())
            ax.set_yticks(())

        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        if not os.path.exists('img'): os.makedirs('img')
        plt.savefig('img/'+name+'_layer_'+str(i+1)+'.png',bbox_inches='tight')
        plt.close(fig)
    
def run_sess(classifier,datasets,filename,load_saver=''):
    X_train, Y_train, X_test, Y_test = datasets
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) # 初始化变量
    if classifier.tbd: summ = Summaries(filename,sess=sess)
    else: summ=None
    classifier.train_model(train_X = X_train, 
                           train_Y = Y_train, 
                           val_X = X_test, 
                           val_Y = Y_test,
                           sess=sess,
                           summ=summ,
                           load_saver=load_saver)
    
    classifier.show_result(filename)
    classifier.save_result()
    if classifier.tbd: 
        summ.train_writer.close()
        
    sess.close()
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

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

class Activation(object):   
    def get_act_func(self,func_name):
        if func_name=='sigmoid': # S(z) = 1/(1+exp(-z)) ∈ (0,1)
            return tf.nn.sigmoid
        if func_name=='softmax': # s(z) = S(z)/∑S(z) ∈ (0,1)
            return tf.nn.softmax
        if func_name=='relu':    # r(z) = max(0,z) ∈ (0,+inf)
            return tf.nn.relu
        
class Loss(object):
    def __init__(self,
                 label_data,
                 pred,
                 output_act_func):
        self.label_data = label_data
        self.pred = pred
        self.output_act_func = output_act_func  
        
    def get_loss_func(self,func_name):
        if func_name=='cross_entropy':
            if self.output_act_func=='softmax':
                return tf.losses.softmax_cross_entropy(self.label_data, self.pred)
            if self.output_act_func=='sigmoid':
                return tf.losses.sigmoid_cross_entropy(self.label_data, self.pred)
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
            pre_lables=tf.floor(self.pred+0.5)
            data_lables=tf.floor(self.label_data+0.5)
        return tf.reduce_mean(tf.cast(tf.equal(pre_lables,data_lables),tf.float32))
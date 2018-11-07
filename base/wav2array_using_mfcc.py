# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:00:37 2018

@author: Administrator
"""
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import sys
import keras

# 注意：把下载好的 wav 文件解压到 wav_dir 这个文件夹, 也可以自己设定别的路径
#      如果要自己做语音的数据集, 需要 pip install librosa
wav_dir = '../../DataSet/Urban Sound Classification'
csv_dir = '../dataset/Urban Sound Classification'

def example(ID):
    file_name =  os.path.abspath(wav_dir+'/Train/'+str(ID)+'.wav')
    print(file_name)
    data, sampling_rate = librosa.load(file_name)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)

def com_stat(w,f,u,s):
    n = f.shape[0]
    m = f.shape[1]
    def skewness(x,u_,s_):
        return np.sum((x-u_)**3)/(m*(m/(m-1))**1.5*s_**3)
    def kurtosis(x,u_,s_):
        return np.sum((x-u_)**4)/(m*s_**4) - 3
    
    X = list()
    for i in range(n):
        if w == 'skewness':
            X.append( skewness(f[i],u[i],s[i]) )
        elif w == 'kurtosis':
            X.append( kurtosis(f[i],u[i],s[i]) )
    X = np.array(X)
    return X  

def make_data(meth='harm', stat= True, dur = 4, sr = 22050, n_feature = 64):
    
    str_='('+meth+').csv'
    
    def parser(ID):
        """
            function to load files and extract features
        """

        file_name = os.path.join(os.path.abspath(wav_dir), tp, str(ID) + '.wav') # load 训练集/测试集样本
        
        X, sample_rate = librosa.load(file_name,sr=sr, res_type='kaiser_fast')
        if stat == False:
            t = librosa.get_duration(filename=file_name)
            if t != dur:
                rate = t/dur *0.99
                X = librosa.effects.time_stretch(X,rate)[:sample_rate*dur]
                if X.shape[0] != sample_rate*dur:
                    print('Warning: X has a sahpe with: ' + str(X.shape[0]) + ', while t = ' + str(t))
        
        if meth=='harm':
            X = librosa.effects.harmonic(y=X, margin=8)
            f = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_feature)
        else:
            f = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_feature)
        
        if stat: ###########################################################################################
            # 对 T 取统计量
            f_mean = np.mean(f,axis=1)
            f_min = np.min(f,axis=1)
            f_max = np.max(f,axis=1)
            f_var = np.var(f,axis=1)
            
            f_skewness = com_stat('skewness',f,f_mean,f_var)
            f_kurtosis = com_stat('kurtosis',f,f_mean,f_var)
            feature = np.concatenate((f_mean,f_min,f_max,f_var,f_skewness,f_kurtosis),axis=0)
            
#            feature = np.concatenate((f_mean,f_min,f_max,f_var),axis=0)
            
        else: #########################################################################################
            feature = f  #  (n_feature, T)
            
        # console 动态文本显示
        sys.stdout.write('\r'+"Processing audio files: ["+ tp + '] ' + str(ID) + '.wav' + ' - ' + str(feature.shape))
        sys.stdout.flush()

        return feature

    def make_data_set(tp):

        from sklearn.preprocessing import LabelEncoder

        print(tp + " DATA...")
        ID_Class = pd.read_csv(os.path.join(os.path.abspath(csv_dir), tp+'.csv')) # load 训练集/测试集标签序号ID
        
        feature = list()
        for i in range(len(ID_Class)):
            f = parser(ID_Class.iat[i,0]).reshape(1,-1)
            
            if i==0: 
                sp = f.shape
            elif f.shape != sp:
                f = f[:,:sp[1]]
                if f.shape != sp:
                    print('Warning: f has a sahpe with: ' + str(f.shape) + ', while t = ' + str(i))
                    
            feature.append(f)
        X = np.concatenate(feature,axis = 0)
        
        print(" --> finished!")
        
        print(X.shape)
        
        np.savetxt(os.path.join(os.path.abspath(csv_dir),tp+'_X'+str_),np.array(X,dtype=np.float32),     # save 训练集样本
                   fmt='%.6f',delimiter=",")
        if tp == 'Train':
            y = np.array(ID_Class.Class.tolist())
            lb = LabelEncoder()
            y = keras.utils.np_utils.to_categorical(lb.fit_transform(y))
            np.savetxt(os.path.join(os.path.abspath(csv_dir),tp+'_Y'+str_),np.array(y,dtype=np.int), # save 训练集标签
                       fmt='%d',delimiter=",")

            print(ID_Class.Class.value_counts())
            print(lb.classes_)
            np.savetxt(os.path.join(os.path.abspath(csv_dir),'label_encoder.csv'),lb.classes_, # save 标签转换格式
                       fmt='%s',delimiter=",")
            return X,y
        else:
            return X
    
    tp = 'Train'
    train_X, train_Y = make_data_set(tp)
    tp = 'Test'
    test_X = make_data_set(tp)
    
    return train_X, train_Y, test_X

#####################################################################################################################

if __name__ == "__main__":
    method = 2
    if method == 1: # mfcc
        dataset = make_data(meth='mfcc', stat= True, dur = 4, sr = 44100, n_feature = 64)
    elif method == 2: # harm
        dataset = make_data(meth='harm', stat= True, dur = 4, sr = 44100, n_feature = 64)
    example(0)

############
#    包含网络    #
############

——> 推荐使用：
Deep Belief Network (DBN) 
Stacked Autoencoder (sAE) 
Stacked Sparse Autoencoder (sSAE) 
Stacked Denoising Autoencoders (sDAE) 
——> 尝试更好的模型：
Convolutional Neural Network (CNN) 
Recurrent Neural Network (RNN) 
Long Short Term Memory (LSTM) 

############
#    所依赖包    #
############

pip install tensorflow
pip install keras
pip install librosa （用于语音分类，选装）
pip install --upgrade numpy pandas（有问题一般是由于你的包需要升级了）

############
#    版本信息    #
############

Note 用户可以通过model.py文件控制一些功能的开关： 
·→ self.show_pic =True       # show curve in 'Console'?
·→ self.tbd = False              # open/close tensorboard
·→ self.save_model = False  # save/ not save model
·→ self.plot_para=False       # plot W image or not
·→ self.save_weight = False # save W matrix or not
·→ self.do_tSNE = False       # do t-SNE or not

Version 2018.11.7
New 新增了两个数据集，一个用于分类，一个用于预测
New 新增t-SNE低维可视化
Chg 修正部分 use_for = 'prediction' 时的Bug

Version 2018.6.1 
New 新增了绘制训练曲线图，预测标签分布图，权值图的功能 
Chg 重写了SAE，现在可以放心使用了 
Chg 代码的整体运行函数run_sess放到了base_func.py 
Chg 回归是可以实现的，需要设置 use_for = 'prediction' 

############
#    测试结果   #
############

用于minst数据集分类，运行得到正确率可达98.78%；
用于Urban Sound Classification语音分类，正确率达73.37%；
用于Big Mart Sales III预测，RMSE为1152.04

跑的结果并不是太高，有更好的方法请赐教。
语音分类未尝试语谱法，欢迎做过的和我交流。

############
#    参考资料  #
############

TF基本函数：http://www.cnblogs.com/wuzhitj/p/6431381.html 
RBM原理：https://blog.csdn.net/itplus/article/details/19168937 
Hinton源码：http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html 
sDAE原论文：http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf 
sSAE分析TE：https://www.sciencedirect.com/science/article/pii/S0169743917302496 
RNN原理：https://zhuanlan.zhihu.com/p/28054589 
LSTM：https://www.jianshu.com/p/9dc9f41f0b29 
Tensorboard：https://blog.csdn.net/sinat_33761963/article/details/62433234 

############
#    My blog  #
############

知乎：https://www.zhihu.com/people/fu-zi-36-41/posts 
CSDN：https://blog.csdn.net/fuzimango/article/list/ 
QQ群：640571839 
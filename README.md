# 包含网络
——> 推荐使用：<br />
Deep Belief Network (DBN) <br />
Stacked Autoencoder (sAE) <br />
Stacked Sparse Autoencoder (sSAE) <br />
Stacked Denoising Autoencoders (sDAE) <br />
——> 尝试更好的模型：<br />
Convolutional Neural Network (CNN) <br />
Recurrent Neural Network (RNN) <br />
Long Short Term Memory (LSTM) <br />

# 所依赖包
pip install tensorflow <br />
pip install keras <br />
pip install librosa （用于语音分类，选装） <br />
pip install --upgrade numpy pandas（升级包） <br />

# 版本信息
Note 用户可以通过model.py文件控制一些功能的开关： <br />
·→ self.show_pic: show curve in 'Console'? <br />
·→ self.tbd: open/close tensorboard <br />
·→ self.save_model: save/ not save model <br />
·→ self.plot_para: plot W image or not <br />
·→ self.save_weight: save W matrix or not <br />
·→ self.do_tSNE: do t-SNE or not <br />
<br />
Version 2018.11.7 <br />
New 新增了两个数据集，一个用于分类，一个用于预测 <br />
New 新增t-SNE低维可视化 <br />
Chg 修正部分 use_for = 'prediction' 时的Bug <br />
<br />
Version 2018.6.1  <br />
New 新增了绘制训练曲线图，预测标签分布图，权值图的功能 <br />
Chg 重写了SAE，现在可以放心使用了 <br />
Chg 代码的整体运行函数run_sess放到了base_func.py <br />
Chg 回归是可以实现的，需要设置 use_for = 'prediction' <br />

# 测试结果
用于minst数据集分类，运行得到正确率可达98.78%；<br />
用于Urban Sound Classification语音分类，正确率达73.37%；<br />
https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/ <br />
用于Big Mart Sales III预测，RMSE为1152.04 <br />
https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/ <br />
<br />
跑的结果并不是太高，有更好的方法请赐教。 <br />
语音分类未尝试语谱法，欢迎做过的和我交流。<br />

# 参考资料
TF基本函数：http://www.cnblogs.com/wuzhitj/p/6431381.html <br />
RBM原理：https://blog.csdn.net/itplus/article/details/19168937 <br />
Hinton源码：http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html <br />
sDAE原论文：http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf <br />
sSAE分析TE：https://www.sciencedirect.com/science/article/pii/S0169743917302496 <br />
RNN原理：https://zhuanlan.zhihu.com/p/28054589 <br />
LSTM：https://www.jianshu.com/p/9dc9f41f0b29 <br />
Tensorboard：https://blog.csdn.net/sinat_33761963/article/details/62433234 <br />

# My blog
知乎：https://www.zhihu.com/people/fu-zi-36-41/posts <br />
CSDN：https://blog.csdn.net/fuzimango/article/list/ <br />
QQ群：640571839 <br />

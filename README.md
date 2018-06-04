# 包含网络
Deep Belief Network (DBN) <br />
Convolutional Neural Network (CNN) <br />
Recurrent Neural Network (RNN) <br />
Long Short Term Memory (LSTM) <br />
Stacked Autoencoder (sAE) <br />
Stacked Sparse Autoencoder (sSAE) <br />
Stacked Denoising Autoencoders (sDAE) <br />
# 版本信息
Version 2018.6.1 <br />
Chg 重写了SAE，现在可以放心使用了 <br />
Chg 新增了绘制训练曲线图，预测标签分布图，权值图的功能 <br />
Chg 代码的整体运行函数run_sess放到了base_func.py <br />
N 用户可以通过model.py文件控制一些功能的开关： <br />
·→  tensorboard(self.tbd) <br />
·→  saver(self.sav) <br />
·→  显示曲线(self.show_pic) <br />
·→  保存权值图(self.plot_para) <br />
# 测试结果
用于minst数据集分类，运行得到正确率可达98.78%； <br />
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
p.s.：有Bug请向我反馈 <br />

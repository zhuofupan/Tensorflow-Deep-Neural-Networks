# 包含网络
- 推荐使用: </br>
*Deep Belief Network (DBN)*  </br>
*Stacked Autoencoder (sAE)* </br>
*Stacked Sparse Autoencoder (sSAE)* </br>
*Stacked Denoising Autoencoders (sDAE)* </br>
- 尝试更好的模型：</br>
*Convolutional Neural Network (CNN)* </br>
*Recurrent Neural Network (RNN)* </br>
*Long Short Term Memory (LSTM)* </br>
# 所依赖包
```python
pip install tensorflow (version: 1.X)
pip install keras
pip install librosa (用于语音分类，选装)
pip install --upgrade --user numpy pandas h5py (升级包)
```
# 用于任务
`use_for = 'classification'` 用于分类任务 </br>
`use_for = 'prediction'` 用于预测任务 </br>

# 版本信息
## Pytorch版本：
推荐[PyTorch包](https://github.com/fuzimaoxinan/torch-fuzz) </br>

## User：
用户可以通过`model.py`文件控制一些功能的开关： </br>
`self.show_pic` => show curve in 'Console'? </br>
`self.tbd` => open/close tensorboard </br>
`self.save_model` => save/ not save model </br>
`self.plot_para` => plot W image or not </br>
`self.save_weight` => save W matrix or not </br>
`self.do_tSNE` => do t-SNE or not

## Version 2018.11.7:
New 新增了两个数据集，一个用于分类，一个用于预测 </br>
New 新增t-SNE低维可视化 </br>
Chg 修正部分 `use_for = 'prediction'` 时的Bug

## Version 2018.6.1:
New 新增了绘制训练曲线图，预测标签分布图，权值图的功能 </br>
Chg 重写了SAE，现在可以放心使用了 </br>
Chg 代码的整体运行函数`run_sess`放到了`base_func.py` </br>
Chg 回归是可以实现的，需要设置 `use_for = 'prediction'`

# 测试结果
用于`minst`数据集分类，运行得到正确率可达98.78% </br>
用于`Urban Sound Classification`语音分类，正确率达73.37% </br>
(这个跑完console不会显示结果，因为是网上的比赛数据集，需上传才能得到正确率)</br>
用于`Big Mart Sales III`预测，RMSE为1152.04 </br>
(这个也是网上的数据集，也没有test_Y)</br></br>

跑的结果并不是太高，有更好的方法请赐教 </br>
语音分类未尝试语谱法，欢迎做过的和我交流 </br>

# 数据地址
[USC](https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/), 
[BMS III](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/) 

# 参考资料
[Tensorflow基本函数](http://www.cnblogs.com/wuzhitj/p/6431381.html), 
[RBM原理](https://blog.csdn.net/itplus/article/details/19168937), 
[Hinton源码](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html), 
[sDAE原论文](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf), 
[sSAE分析TE过程](https://www.sciencedirect.com/science/article/pii/S0169743917302496), 
[RNN原理](https://zhuanlan.zhihu.com/p/28054589), 
[LSTM](https://www.jianshu.com/p/9dc9f41f0b29), 
[Tensorboard](https://blog.csdn.net/sinat_33761963/article/details/62433234) 

# My blog
[ResearchGate](https://www.researchgate.net/profile/Zhuofu-Pan),
[知乎](https://www.zhihu.com/people/fu-zi-36-41/posts), 
[CSDN](https://blog.csdn.net/fuzimango/article/list/) </br>
QQ群：640571839

# Paper
希望大家多支持支持我们的工作，欢迎交流探讨~</br>
[1] Z. Pan, H. Chen, Y. Wang, B. Huang, and W. Gui, "[A new perspective on ae-and vae-based process monitoring](https://www.techrxiv.org/articles/preprint/A_New_Perspective_on_AE-_and_VAE-based_Process_Monitoring/19617534)," TechRxiv, Apr. 2022, doi.10.36227/techrxiv.19617534. </br>
[2] Z. Pan, Y. Wang, k. Wang, G. Ran, H. Chen, and W. Gui, "[Layer-Wise Contribution-Filtered Propagation for Deep Learning-Based Fault Isolation](https://onlinelibrary.wiley.com/doi/10.1002/rnc.6328)," Int. J. Robust Nonlinear Control, Jul. 2022, doi.10.1002/rnc.6328 </br>
[3] Z. Pan, Y. Wang, K. Wang, H. Chen, C. Yang, and W. Gui, "[Imputation of Missing Values in Time Series Using an Adaptive-Learned Median-Filled Deep Autoencoder](https://ieeexplore.ieee.org/document/9768200)," IEEE Trans. Cybern., 2022, doi.10.1109/TCYB.2022.3167995 </br>
[4] Y. Wang, Z. Pan, X. Yuan, C. Yang, and W. Gui, "[A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub),” ISA Trans., vol. 96, pp. 457–467, 2020. </br>
[5] Z. Pan, Y. Wang, X. Yuan, C. Yang, and W. Gui, "[A classification-driven neuron-grouped sae for feature representation and its application to fault
classification in chemical processes](https://www.sciencedirect.com/science/article/pii/S0950705121006122) ," Knowl.-Based Syst., vol. 230, p. 107350, 2021. </br>
[6] H. Chen, B. Jiang, S. X. Ding, and B. Huang, "[Data-driven fault diagnosis for traction systems in high-speed trains: A survey, challenges, and perspectives](https://ieeexplore.ieee.org/abstract/document/9237134?casa_token=s5x0G5FMme0AAAAA:DuVqfDrkdk06Vgzx_mw1LW-QRVTHMje-3Yvf8p8-uoRPLIft02J48fkObFy_tj0yHznVbYKu)," IEEE Trans. Intell. Transp. Syst., 2020, doi.10.1109/TITS.2020.3029946 </br>
[7] H. Chen and B. Jiang, "[A review of fault detection and diagnosis for the traction system in high-speed trains](https://ieeexplore.ieee.org/abstract/document/8654208?casa_token=GLx56ooTyeAAAAAA:Csb_mMFIUGBAqbs30ozzKZMC9OYlT4klWmC-m9Xa_qIjuezRaA6kqpAGqjfugGbYIWtPFZYW)," IEEE Trans. Intell. Transp. Syst., vol. 21, no. 2, pp. 450–465, Feb. 2020. </br>


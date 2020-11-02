##  BN（Batch Normalization）层的作用

（1）加速收敛
（2）控制过拟合，可以少用或不用Dropout和正则
（3）降低网络对初始化权重不敏感
（4）允许使用较大的学习率

LN提出：BN针对一个minibatch的输入样本，计算均值和方差，
基于计算的均值和方差来对某一层神经网络的输入X中每一个case进行归一化操作。
但BN有两个明显不足：
BN与LN的不同之处在于：LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
而BN中则针对不同神经元输入计算均值和方差，同一个minibatch中的输入拥有相同的均值和方差。
因此，LN不依赖于mini-batch的大小和输入sequence的深度，
因此可以用于bath-size为1和RNN中对边长的输入sequence的normalize操作。

BN与LN的不同之处在于：LN中同层神经元输入拥有相同的均值和方差，
不同的输入样本有不同的均值和方差；而BN中则针对不同神经元输入计算均值和方差，
同一个minibatch中的输入拥有相同的均值和方差。
因此，LN不依赖于mini-batch的大小和输入sequence的深度，
因此可以用于bath-size为1和RNN中对边长的输入sequence的normalize操作。

BN 和 LN 均将规范化应用于输入的特征数据 ,而 WN 则另辟蹊径,
将规范化应用于线性变换函数的权重 ,这就是 WN 名称的来源。
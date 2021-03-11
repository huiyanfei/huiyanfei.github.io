##  BN（Batch Normalization）层的作用

（1）加速收敛
（2）控制过拟合，可以少用或不用Dropout和正则
（3）降低网络对初始化权重不敏感
（4）允许使用较大的学习率

第一步：

/miu 是平移参数（shift parameter）， /sigma 是缩放参数（scale parameter）。
通过这两个参数进行 shift 和 scale 变换：
    
    x'=(x-/miu)/sigma
    
得到的数据符合均值为 0、方差为 1 的标准分布。
第二步：
b是再平移参数（re-shift parameter），g是再缩放参数（re-scale parameter）。
将 上一步得到的 进一步变换为：

    y=g*x'+b

最终得到的数据符合均值为 b、方差为 g^2 的分布。
为了保证模型的表达能力不因为规范化而下降。

BN针对一个minibatch的输入样本，计算均值和方差，
基于计算的均值和方差来对某一层神经网络的输入X中每一个case进行归一化操作。
但BN有两个明显不足：
1、高度依赖于mini-batch的大小，实际使用中会对mini-Batch大小进行约束，
不适合类似在线学习（mini-batch为1）情况；
2、不适用于RNN网络中normalize操作：BN实际使用时需要计算并且保存
某一层神经网络mini-batch的均值和方差等统计信息，
对于对一个固定深度的前向神经网络（DNN，CNN）使用BN，很方便；
但对于RNN来说，sequence的长度是不一致的，换句话说RNN的深度不是固定的，
不同的time-step需要保存不同的statics特征，
可能存在一个特殊sequence比其的sequence长很多，这样training时，计算很麻烦。
 


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

## LN
# Introduction-to-AI

This is a basic implementation of Artificial intellengence.

week 13：[How to evaluate the performance including regression and classification](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%201%20Week%2013.ipynb)

[sloution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%201%20Week%2013%20Answers.ipynb)



week 14: [k-nearest neighbours, linear regression, and the naive Bayes classifier.](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%202%20Week%2014.ipynb)

[solution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%202%20Week%2014%20Answers.ipynb)



week 15: [unsupervised clustering algorithms](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%203%20Week%2015.ipynb)

[solution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%203%20Week%2015%20Answers(1).ipynb)

[TOC]

# 无监督学习



## 常见算法



### K聚类算法

#### 过程

1. 随机选取k个样本作为聚类中心
2. 计算各个样本与各个聚类之间的欧式距离
3. 将各样本回归于预知最近的聚类中心
4. 求各个类之间的样本的均值，作为新的聚类中心。（通过求误差平方和的方式）
5. 判定：若类中心不再发生变动或达到迭代次数，算法结束，否则回到第二步



定K——随意定质心——计算点心距离——回归聚类中心——计算新聚类中心——判定



#### 原理

在合理范围内，使得误差平方和最小或者组内平方和最小。



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220145200922.png/" alt="image-20220220145200922" style="zoom:50%;" />



#### K值

可以使用手肘法（The Elbow Method）：

在合理范围内，对WCSS组内平方和生成的图像进行选点

K设置的越大，样本划分越细致，每个聚类聚合程度越高，WCSS组内平方和越小

K设置的越小，样本划分越梳离，每个聚类聚合程度越低，WCSS组内平方和越大



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220145836234.png" alt="image-20220220145836234" style="zoom:30%;" />

​                                                                            如图，2和4之间的点利用手肘法即可设置为K。



#### 要求

K—means 聚类方法只能用于连续数据，不能用于分类数据



#### 预处理数据

1. 需要清洗异常值和outlier
2. 需先对数据进行归一化处理



#### 优缺点

| K-means模型 |                                                              |
| ----------- | ------------------------------------------------------------ |
| 优点        | 1. 无监督学习，不需要标签信息  2. 逻辑简单，可解释性好，易理解 3. 分类效果不错 4. 需要调参的参数只有k这一个，舒服！ |
| 缺点        | 1. 准确度不如监督学习 2. 对K值得选择较为敏感，而k需要人定 3. 由于迭代多，数据多时计算量大，耗时 4. 不适合太离散、太不平衡 、非凸数据分类 |



#### 代码模版

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('11_K_Means/Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values #这里没有因变量 ， 因此叫无监督学习

# Using the elbow method手肘法 to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++",random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Applying the k-means to the mall dataset
kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, init="k-means++",random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100, c="red",label= "Careful" )
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100, c="blue",label= "Standard" )
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s=100, c="green",label= "Target" )
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1],s=100, c="cyan",label= "Careless" )
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1],s=100, c="pink",label= "Sensible" )

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300, c="yellow",label= "Centroids" )
plt.title("Clusters of cliens")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
```



### 分层聚类 （Hierarchical Clustering）



层次聚类算法分为分裂和聚集



#### 分裂算法

在divisive 或者从上至下的聚类方法中，所有的观测值被分配给了一个单一的聚类，然后将这个聚类分开至至少两个相似的聚类. 最后我们在每个聚类上进行递归，直到每个observation只有一个聚类。在某些情况下，分类算法比聚集算法产生的精度更高，但是从概念上更复杂。



#### 聚集算法

在聚集或者自下而上的聚类方法中，把每个观测值分类到他自己的聚类中, 然后计算每个聚类之间的相似度（距离），并且结合两个最相似的聚类，最后，重复步骤直到剩下单一聚类。

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220153051158.png/" alt="image-20220220153051158" style="zoom:33%;" />

回忆一下讲课内容，有很多方法可以测量群组之间的距离。比如说。

* Minimum distance: $d(S,T) = \min\{d(x,y) : x \in S,y \in T\} $
* Average distance: $d(S,T) = \frac{1}{|S||T|} \sum_{(x, y)} d(x, y)$
* Maximum distance: $d(S,T) = \max\{d(x,y) : x \in S,y \in T\} $
* Centroid distance: $ d(S,T) = d(\frac{\sum_{x\in S} x}{|S|} \frac{\sum_{y\in T} y}{|T|})$

```python
from scipy.cluster.hierarchy import dendrogram, linkage

## The following will generate a dendogram for the iris data set:
linked = linkage(X, 'single')
labelList = range(len(X))
plt.figure(figsize=(10, 7)) 
dendrogram(linked,labels=labelList) 
plt.show()

fig, axs=plt.subplots(2,2, figsize=(15, 15))
metrics=['single', 'average', 'complete',  'centroid']
for i in range(2):
    for j in range(2):
        linked=linkage(X, metrics[(i*2)+j])
        dendrogram(linked, ax=axs[i,j])
        axs[i,j].set_title(metrics[(i*2)+j])
        
```





### 高斯混合聚类（Gaussian Mixture Mode，GMM） 



输入：样本集![[公式]](https://www.zhihu.com/equation?tex=D%3D%5Cleft+%5C%7B+x_%7B1%7D%2Cx_%7B2%7D%2C...%2Cx_%7Bm%7D%5Cright+%5C%7D)高斯混合成分个数![[公式]](https://www.zhihu.com/equation?tex=k)

1. 初始化K个多元高斯分布以及其权重
2. 根据贝叶斯定理，估计每个样本由每个成分生成的后验概率；(EM方法中的E步)
3. 根据均值，协方差的定义以及2步求出的后验概率，更新均值向量，协方差矩阵式和权重；（EM方法的M步）
4. 重复2~3步，直到似然函数增加值已小于收敛阈值，或达到最大迭代次数
5. 对于每一个样本点，根据贝叶斯定理计算出其属于每一个簇的后验概率，并将样本划分到后验概率最大的簇上去



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220155351957.png" alt="image-20220220155351957" style="zoom:20%;" />

```python
# Import the iris dataset, and save the data into a variable X (take a look at the documentation here: 
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
from sklearn import datasets
iris = datasets.load_iris() 
X = iris.data

# Gaussian Mixture models
# In this question we investigate the use of Gaussian clustering on the Iris data set.
from sklearn.mixture import GaussianMixture as GMM
gmm = GMM(n_components=3)
gmm.fit(X)

# We can extract the parameters for the learnt Gaussian distributions as follows:
print(gmm.means_)
print(gmm.covariances_)


# How do the means for the three distributions compare with the centroids from a 3-cluster $k$-means on this dataset? 
# A: One of the means is identical: that of the more isolated cluster. The others are close.
# Fit the iris dataset
kmm = KMeans(n_clusters = 3)
kmm.fit(X)
print(kmm.cluster_centers_)
# Make a scatter plot of the data on the first two axes
# Experiment with looking at different axes
plt.figure()
plt.scatter(X[:,2],X[:,3], c=kmm.labels_, cmap='rainbow')



# Use the command `print(gmm.weights_)` to look at the weights for each distribution. What do these weights tell us about the composition of the three clusters?
# A: The weight of the isolated cluster is 1/3, indicating that this cluster accounts for 1/3 of the points. The other clusters each account for just under and just over 1/3 of the points.
print(gmm.weights_)
```

week 16: [Neural Networks](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%204%20Week%2016.ipynb)

[solution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%204%20Week%2016%20answers.ipynb)

# 深度学习

## 神经网络（Artificial Neural Network）

1. 神经元（Artificial Neural）： 神经元是神经网络的基础·



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220161158959.png/" alt="image-20220220161158959" style="zoom:30%;" />



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220161301695.png" alt="image-20220220161301695" style="zoom:30%;" />



2. 单层神经网络

在输入层和输出层之间加入了隐藏层

3. 多层神经网络（MLP）

在输入层和输出层之间加入了多个隐藏层

4. 深度神经网络（DNN）

在输入层和输出层之间加入了很多个隐藏层

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220162546223.png/" alt="image-20220220162546223" style="zoom:30%;" />

5. 网络训练（Loss Function）

首先通过目前的参数进行预测，然后对其预测结果的正确性进行反馈，然后根据反馈更新参数



​                                                 分类问题可以通过通过损失函数来进行操作

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220162943256.png" alt="image-20220220162943256" style="zoom:25%;" />





<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220163013631.png" alt="image-20220220163013631" style="zoom:25%;" />

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220163418652.png" alt="image-20220220163418652" style="zoom:25%;" />

概率问题可以通过Softmax 函数进行操作

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220163729874.png" alt="image-20220220163729874" style="zoom:80%;" />

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220164107653.png/" alt="image-20220220164107653" style="zoom:80%;" />

### [卷积神经网络（CNN）](https://zhuanlan.zhihu.com/p/47184529)

CNN不需要预处理图像

卷积神经网络主要由这几类层构成：输入层、卷积层，ReLU层、池化（Pooling）层和全连接层（全连接层和常规神经网络中的一样）。通过将这些层叠加起来，就可以构建一个完整的卷积神经网络。在实际应用中往往将卷积层与ReLU层共同称之为卷积层，**所以卷积层经过卷积操作也是要经过激活函数的**。具体说来，卷积层和全连接层（CONV/FC）对输入执行变换操作的时候，不仅会用到激活函数，还会用到很多参数，即神经元的权值w和偏差b；而ReLU层和池化层则是进行一个固定不变的函数操作。卷积层和全连接层中的参数会随着梯度下降被训练，这样卷积神经网络计算出的分类评分就能和训练集中的每个图像的标签吻合了。

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220164722277.png" alt="image-20220220164722277" style="zoom:30%;" />

卷积神经网络最重要的层是卷积层 ，他可以是多维的，并且可以捕捉到相邻数据点的关联性

#### 卷积层作用

1. **滤波器的作用或者说是卷积的作用**。卷积层的参数是有一些可学习的滤波器集合构成的。每个滤波器在空间上（宽度和高度）都比较小，**但是深度和输入数据一致**（这一点很重要，后面会具体介绍）。直观地来说，网络会让滤波器学习到当它看到某些类型的视觉特征时就激活，具体的视觉特征可能是某些方位上的边界，或者在第一层上某些颜色的斑点，甚至可以是网络更高层上的蜂巢状或者车轮状图案。

2. **可以被看做是神经元的一个输出**。神经元只观察输入数据中的一小部分，并且和空间上左右两边的所有神经元共享参数（因为这些数字都是使用同一个滤波器得到的结果）。

3. **降低参数的数量**。这个由于卷积具有“权值共享”这样的特性，可以降低参数数量，达到降低计算开销，防止由于参数过多而造成过拟合。

#### 池化层

通常在连续的卷积层之间会周期性地插入一个池化层。它的作用是逐渐降低数据体的空间尺寸，这样的话就能减少网络中参数的数量，使得计算资源耗费变少，也能有效控制过拟合。汇聚层使用 MAX 操作，对输入数据体的每一个深度切片独立进行操作，改变它的空间尺寸。最常见的形式是汇聚层使用尺寸2x2的滤波器，以步长为2来对每个深度切片进行降采样，将其中75%的激活信息都丢掉。每个MAX操作是从4个数字中取最大值（也就是在深度切片中某个2x2的区域），深度保持不变。

#### 归一化层

在卷积神经网络的结构中，提出了很多不同类型的归一化层，有时候是为了实现在生物大脑中观测到的抑制机制。但是这些层渐渐都**不再流行**，因为实践证明**它们的效果即使存在，也是极其有限的**。

#### 全连接层

这个常规神经网络中一样，它们的激活可以先用矩阵乘法，再加上偏差。

week 17: [Decision Trees](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%205%20Week%2017.ipynb)

# 决策树

## 决策树的结构





<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220222411031.png" alt="image-20220220222411031" style="zoom:30%;" />



为了获得更好的泛化，需要使用少量节点

##### 如何建立一个决策树：

用训练例子来组成一个小的决策树

选择最典型的特征来分裂成小枝节



![image-20220220224756818](/Users/wentao/Library/Application Support/typora-user-images/image-20220220224756818.png)



![image-20220220225209614](/Users/wentao/Library/Application Support/typora-user-images/image-20220220225209614.png)



决策树算法：

![image-20220220225550309](/Users/wentao/Library/Application Support/typora-user-images/image-20220220225550309.png)



决策树的优劣衡量：

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220230149729.png" alt="image-20220220230149729" style="zoom:67%;" />





#### 决策树优化（剪枝）



<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220231007635.png/" alt="image-20220220231007635" style="zoom:67%;" />

<img src="/Users/wentao/Library/Application Support/typora-user-images/image-20220220230809857.png" alt="image-20220220230809857" style="zoom:67%;" />

#### 随机森林：

![image-20220220231441549](/Users/wentao/Library/Application Support/typora-user-images/image-20220220231441549.png)

![image-20220220231535140](/Users/wentao/Library/Application Support/typora-user-images/image-20220220231535140.png)

**下面是随机森林的构造过程：**

  　　1. 假如有N个样本，则有放回的随机选择N个样本(每次随机选择一个样本，然后返回继续选择)。这选择好了的N个样本用来训练一个决策树，作为决策树根节点处的样本。

  　　2. 当每个样本有M个属性时，在决策树的每个节点需要分裂时，随机从这M个属性中选取出m个属性，满足条件m << M。然后从这m个属性中采用某种策略（比如说信息增益）来选择1个属性作为该节点的分裂属性。

  　　3. 决策树形成过程中每个节点都要按照步骤2来分裂（很容易理解，如果下一次该节点选出来的那一个属性是刚刚其父节点分裂时用过的属性，则该节点已经达到了叶子节点，无须继续分裂了）。一直到不能够再分裂为止。注意整个决策树形成过程中没有进行剪枝。

  　　4. 按照步骤1~3建立大量的决策树，这样就构成了随机森林了。

在建立每一棵决策树的过程中，有两点需要注意采样与完全分裂。

首先是两个随机采样的过程，random forest对输入的数据要进行行、列的采样。对于行采样，采用有放回的方式，也就是在采样得到的样本集合中，可能有重复的样本。假设输入样本为N个，那么采样的样本也为N个。这样使得在训练的时候，每一棵树的输入样本都不是全部的样本，使得相对不容易出现over-fitting。然后进行列采样，从M个feature中，选择m个（m << M）。

之后就是对采样之后的数据使用完全分裂的方式建立出决策树，这样决策树的某一个叶子节点要么是无法继续分裂的，要么里面的所有样本的都是指向的同一个分类。一般很多的决策树算法都一个重要的步骤——剪枝，但是这里不这样干，由于之前的两个随机采样的过程保证了随机性，所以就算不剪枝，也不会出现over-fitting。

通过分类，子集合的熵要小于未分类前的状态，这就带来了信息增益（information gain）



**决策树有很多的优点：**

a. 在数据集上表现良好，两个随机性的引入，**使得随机森林不容易陷入过拟合**

b. 在当前的很多数据集上，相对其他算法有着很大的优势，两个随机性的引入，使得随机森林具有**很好的抗噪声能力**

c. 它能够**处理很高维度（feature很多）的数据，并且不用做特征选择，**对数据集的适应能力强：既能处理离散型数据，也能处理连续型数据，数据集无需规范化

d. 可生成一个Proximities=（pij）矩阵，用于**度量样本之间的相似性**： pij=aij/N, aij表示样本i和j出现在随机森林中同一个叶子结点的次数，N随机森林中树的颗数

e. 在创建随机森林的时候，对generlization error使用的是**无偏估计**

f. **训练速度快**，可以得到变量重要性排序（两种：基于OOB误分率的增加量和基于分裂时的GINI下降量

g. 在训练过程中，**能够检测到feature间的互相影响**

h. **容易做成并行化方法**

**i. 实现比较简单**



**随机森林主要应用于回归和分类。**本文主要探讨基于随机森林的分类问题。随机森林和使用决策树作为基本分类器的（bagging）有些类似。以决策树为基本模型的bagging在每次bootstrap放回抽样之后，产生一棵决策树，抽多少样本就生成多少棵树，在生成这些树的时候没有进行更多的干预。而随机森林也是进行bootstrap抽样，但它与bagging的区别是：在生成每棵树的时候，每个节点变量都仅仅在随机选出的少数变量中产生。因此，不但样本是随机的，连每个节点变量（Features）的产生都是随机的。

许多研究表明， 组合分类器比单一分类器的分类效果好，**随机森林（random forest）是一种利用多个分类树对数据进行判别与分类的方法，它在对数据进行分类的同时，还可以给出各个变量（基因）的重要性评分，评估各个变量在分类中所起的作用。**

随机森林算法得到的随机森林中的每一棵都是很弱的，但是大家组合起来就很厉害了。我觉得可以这样比喻随机森林算法：每一棵决策树就是一个精通于某一个窄领域 的专家（因为我们从M个feature中选择m让每一棵决策树进行学习），这样在随机森林中就有了很多个精通不同领域的专家，对一个新的问题（新的输入数 据），可以用不同的角度去看待它，最终由各个专家，投票得到结果。而这正是群体智慧（swarm intelligence），经济学上说的看不见的手，也是这样一个分布式的分类系统，由每一自己子领域里的专家，利用自己独有的默会知识，去对一项产品进行分类，决定是否需要生产。随机森林的效果取决于多个分类树要相互独立，要想经济持续发展，不出现overfiting（就是由政府主导的经济增长，但在遇到新情况后产生泡沫），我们就需要要企业独立发展，独立选取自己的feature 。

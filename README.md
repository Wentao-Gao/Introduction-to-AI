# [Introduction-to-AI](https://gwt9970161.github.io/Introduction-to-AI/)

<img src="https://user-images.githubusercontent.com/77952995/155898164-e655ad89-b18a-44c7-9a45-55872938d6d1.png" width="100%">


[Week 13](#week-13-How-to-evaluate-the-performance-including-regression-and-classification)

[Week 14](#week-14-k-nearest-neighbours-linear-regression-and-the-naive-Bayes-classifier)

[Week 15](#week-15-unsupervised-clustering-algorithms)

[Week 16](#week-16-Neural-Networks)

[Week 17](#week-17-Decision-Trees)


This is a basic implementation of Artificial intellengence.

# week 13：[How to evaluate the performance including regression and classification](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%201%20Week%2013.ipynb)

[sloution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%201%20Week%2013%20Answers.ipynb)

建模的评估一般可以分为回归、分类和聚类的评估，本文主要介绍回归和分类的模型评估：

一、回归模型的评估
主要有以下方法：



 
 ```python

指标	                       描述	      metrics方法
Mean Absolute Error(MAE)	平均绝对误差	  from sklearn.metrics import mean_absolute_error
Mean Square Error(MSE)	        平均方差	      from sklearn.metrics import mean_squared_error
R-Squared	                R平方值	       from sklearn.metrics import r2_score

#sklearn的调用

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
 
mean_absolute_error(y_test,y_predict)
mean_squared_error(y_test,y_predict)
r2_score(y_test,y_predict)
```

（一）平均绝对误差（Mean Absolute Error，MAE）

平均绝对误差就是指预测值与真实值之间平均相差多大 ：

![image](https://user-images.githubusercontent.com/77952995/155895371-0ac938b0-343a-4ef9-8fe2-576bbb670dff.png)

平均绝对误差能更好地反映预测值误差的实际情况.

（二）均方误差（Mean Squared Error，MSE）

观测值与真值偏差的平方和与观测次数的比值：

![image](https://user-images.githubusercontent.com/77952995/155895351-ed57a6bc-0ffc-4443-8c2f-719c2c2e83c7.png)

这也是线性回归中最常用的损失函数，线性回归过程中尽量让该损失函数最小。那么模型之间的对比也可以用它来比较。
MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。

（三）R-square(决定系数)

![image](https://user-images.githubusercontent.com/77952995/155895343-40a13447-e829-4576-b7fa-4557c55d7a7a.png)
数学理解： 分母理解为原始数据的离散程度，分子为预测数据和原始数据的误差，二者相除可以消除原始数据离散程度的影响
其实“决定系数”是通过数据的变化来表征一个拟合的好坏。
理论上取值范围（-∞，1], 正常取值范围为[0 1] ------实际操作中通常会选择拟合较好的曲线计算R²，因此很少出现-∞
越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
越接近0，表明模型拟合的越差
经验值：>0.4， 拟合效果好
缺点：数据集的样本越大，R²越大，因此，不同数据集的模型结果比较会有一定的误差

（四）Adjusted R-Square (校正决定系数）

![image](https://user-images.githubusercontent.com/77952995/155895335-65f6879d-7306-4554-a514-fa566db57b10.png)     

n为样本数量，p为特征数量

消除了样本数量和特征数量的影响
（五）交叉验证（Cross-Validation）

交叉验证，有的时候也称作循环估计（Rotation Estimation），是一种统计学上将数据样本切割成较小子集的实用方法，该理论是由Seymour Geisser提出的。在给定的建模样本中，拿出大部分样本进行建模型，留小部分样本用刚建立的模型进行预报，并求这小部分样本的预报误差，记录它们的平方加和。这个过程一直进行，直到所有的样本都被预报了一次而且仅被预报一次。把每个样本的预报误差平方加和，称为PRESS(predicted Error Sum of Squares)。
　　交叉验证的基本思想是把在某种意义下将原始数据(dataset)进行分组,一部分做为训练集(train set)，另一部分做为验证集(validation set or test set)。首先用训练集对分类器进行训练，再利用验证集来测试训练得到的模型(model)，以此来做为评价分类器的性能指标。
　　无论分类还是回归模型，都可以利用交叉验证，进行模型评估，示例代码：

```python
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_cal_score(lr, X, y, cv=2))
```
 

二、分类模型的评估：
准确率、精确率、召回率、f1_score，混淆矩阵，ks，ks曲线，ROC曲线，psi等。

（一）模型准确度评估

1、准确率、精确率、召回率、f1_score

1.1 准确率（Accuracy）的定义是：对于给定的测试集，分类模型正确分类的样本数与总样本数之比；

1.2 精确率（Precision）的定义是：对于给定测试集的某一个类别，分类模型预测正确的比例，或者说：分类模型预测的正样本中有多少是真正的正样本；

1.3 召回率（Recall）的定义为：对于给定测试集的某一个类别，样本中的正类有多少被分类模型预测正确召回率的定义为：对于给定测试集的某一个类别，样本中的正类有多少被分类模型预测正确；

1.4 F1_score，在理想情况下，我们希望模型的精确率越高越好，同时召回率也越高越高，但是，现实情况往往事与愿违，在现实情况下，精确率和召回率像是坐在跷跷板上一样，往往出现一个值升高，另一个值降低，那么，有没有一个指标来综合考虑精确率和召回率了，这个指标就是F值。F值的计算公式为：

    式中：P: Precision， R: Recall, a：权重因子。

当a=1时，F值便是F1值，代表精确率和召回率的权重是一样的，是最常用的一种评价指标。

F1的计算公式为： ![image](https://user-images.githubusercontent.com/77952995/155895298-ecc14166-dca0-43d4-9e6b-740c1a9638b5.png)
代码示例：
  
 ```python
#1、准确率
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,9,9,8,5,8]
y_true = [0, 1, 2, 3,2,6,3,5,9]
 
accuracy_score(y_true, y_pred)
Out[127]: 0.33333333333333331
 
accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均
Out[128]: 3
 
#2、分类报告：输出包括了precision/recall/fi-score/均值/分类个数
 from sklearn.metrics import classification_report
 y_true = [0, 1, 2, 2, 0]
 y_pred = [0, 0, 2, 2, 0]
 target_names = ['class 0', 'class 1', 'class 2']
 print(classification_report(y_true, y_pred, target_names=target_names))
 
#3、特别的对于用predict_proba进行预测计算，那么必须用roc_auc_score，否则会报错
#示例代码
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
 
lr = LogisticRegression(C = 0.0001,class_weight='balanced')   # c 正则化参数
lr.fit(poly_train, target)
lr_poly_pred = lr.predict_proba(poly_test)[:,1]
lr_poly_pred2= lr.predict_proba(poly_train)[:,1]
# submission dataframe
submit = Id.copy()
submit['TARGET'] = lr_poly_pred
 
print('score:',roc_auc_score(target,lr_poly_pred2))
  ```
        
### 1、准确率
建模的评估一般可以分为回归、分类和聚类的评估，本文主要介绍回归和分类的模型评估：

 ```python
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,9,9,8,5,8]
y_true = [0, 1, 2, 3,2,6,3,5,9]
 
accuracy_score(y_true, y_pred)
Out[127]: 0.33333333333333331
 
accuracy_score(y_true, y_pred, normalize=False)  # 类似海明距离，每个类别求准确后，再求微平均
Out[128]: 3
 ```

 
### 2、分类报告：输出包括了precision/recall/fi-score/均值/分类个数
  ```python
 from sklearn.metrics import classification_report
 y_true = [0, 1, 2, 2, 0]
 y_pred = [0, 0, 2, 2, 0]
 target_names = ['class 0', 'class 1', 'class 2']
 print(classification_report(y_true, y_pred, target_names=target_names))
   ```
### 3、特别的对于用predict_proba进行预测计算，那么必须用roc_auc_score，否则会报错
#示例代码
  ```python
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
 
lr = LogisticRegression(C = 0.0001,class_weight='balanced')   # c 正则化参数
lr.fit(poly_train, target)
lr_poly_pred = lr.predict_proba(poly_test)[:,1]
lr_poly_pred2= lr.predict_proba(poly_train)[:,1]
# submission dataframe
submit = Id.copy()
submit['TARGET'] = lr_poly_pred
 
print('score:',roc_auc_score(target,lr_poly_pred2))
  ```
2、混淆矩阵

2.1 基本概念：混淆矩阵也称误差矩阵，是表示精度评价的一种标准格式，用n行n列的矩阵形式来表示。具体评价指标有总体精度、制图精度、用户精度等，这些精度指标从不同的侧面反映了图像分类的精度。

2.1.1 混淆矩阵一级指标（最底层的）：

真实值是positive，模型认为是positive的数量（True Positive=TP）；
真实值是positive，模型认为是negative的数量（False Negative=FN）：这就是统计学上的第一类错误（Type I Error）；
真实值是negative，模型认为是positive的数量（False Positive=FP）：这就是统计学上的第二类错误（Type II Error）；
真实值是negative，模型认为是negative的数量（True Negative=TN）



2.1.2 二级指标

混淆矩阵里面统计的是个数，有时候面对大量的数据，光凭算个数，很难衡量模型的优劣。因此混淆矩阵在基本的统计结果上又延伸了如下4个指标，我称他们是二级指标（通过最底层指标加减乘除得到的）：

准确率（Accuracy）—— 针对整个模型
精确率（Precision）
灵敏度（Sensitivity）：就是召回率（Recall）
特异度（Specificity）


2.1.3 三级指标

F1 Score，公式如下



其中，P代表Precision，R代表Recall。F1-Score指标综合了Precision与Recall的产出的结果。

2.1.4 示例及实现代码

# 假如有一个模型在测试集上得到的预测结果为：  
y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3] # 实际的类别  
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3] # 模型预测的类别  
  # 使用sklearn 模块计算混淆矩阵  
from sklearn.metrics import confusion_matrix  
confusion_mat = confusion_matrix(y_true, y_pred)  
print(confusion_mat) #看看混淆矩阵长啥样  
 
 
[[2 1 0 0]
 
 [0 2 0 0]
 
 [0 0 1 0]
 
 [0 1 0 2]]



2.2 混淆矩阵可视化

```python
def plot_confusion_matrix(confusion_mat):  
    '''''将混淆矩阵画图并显示出来'''  
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)  
    plt.title('Confusion matrix')  
    plt.colorbar()  
    tick_marks = np.arange(confusion_mat.shape[0])  
    plt.xticks(tick_marks, tick_marks)  
    plt.yticks(tick_marks, tick_marks)  
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.show()  
  
plot_confusion_matrix(confusion_mat)  
```

3、ROC曲线和AUC计算

3.1计算ROC值

```python
import numpy as np
from sklearn.metrics import roc_auc_score
 
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)
```



FPR：负正类率（(False Positive Rate），FPR=1-TNR

ROC曲线图



3.2  AUC（Area Under Curve）

AUC就是ROC 曲线下的面积，通常情况下数值介于0.5-1之间，可以评价分类器的好坏，数值越大说明越好。

AUC值是一个概率值，当你随机挑选一个正样本以及一个负样本，当前的分类算法根据计算得到的Score值将这个正样本排在负样本前面的概率就是AUC值。当然，AUC值越大，当前的分类算法越有可能将正样本排在负样本前面，即能够更好的分类。

AUC评价：

  AUC = 1采用这个预测模型时，不管设定什么阈值都能得出完美预测。绝大多数预测的场合，不存在完美分类器。

  0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。

  AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。

  AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测，因此不存在AUC < 0.5的情况。

4、LIft和gain

Lift图衡量的是，与不利用模型相比，模型的预测能力“变好”了多少，lift(提升指数)越大，模型的运行效果越好。
Gain图是描述整体精准度的指标。
计算公式如下：



作图步骤：
（1） 根据学习器的预测结果（注意，是正例的概率值，非0/1变量）对样本进行排序（从大到小）-----这就是截断点依次选取的顺序；
（2） 按顺序选取截断点，并计算Lift和Gain ---也可以只选取n个截断点，分别在1/n，2/n，3/n等位置





（二）模型区分度

金融建模评分卡模型的结果需要能对好、坏人群给出区分，衡量的方法主要有：

（1）好、坏人群的分数（或违约概率）的分布的差异：KS；

（2）好、坏人群的分数（或违约概率）的距离：Divegence；

（3）好、坏人群浓度的差异：Gini。

1、KS值



ks曲线是将每一组的概率的好客户以及坏客户的累计占比连接起来的两条线，ks值是当有一个点，好客户减去坏客户的数量是最大的。那么ks的值的意义在于，我在那个违约概率的点切下去，创造的效益是最高的，就图中这张图来说就是我们大概在第三组的概率的中间的这个概率切下，我可以最大的让好客户进来，会让部分坏客户进来，但是也会有少量的坏客户进来，但是这已经是损失最少了，所以可以接受。那么在建模中是，模型的ks要求是达到0.3以上才是可以接受的。

1.1 KS的计算步骤如下：

（1）计算每个评分区间的好坏账户数；

 （2） 计算每个评分区间的累计好账户数占总好账户数比率(good%)和累计坏账户数占总坏账户数比率(bad%)；

 （3）计算每个评分区间累计坏账户占比与累计好账户占比差的绝对值（累计good%-累计bad%）， 然后对这些绝对值取最大值即得此评分卡的K-S值。

1.2 KS评价：

KS: <20% : 差

KS: 20%-40% : 一般

KS: 41%-50% : 好

KS: 51%-75% : 非常好

KS: >75% : 过高，需要谨慎的验证模型

KS值的不足：ks只能区分度最好的分数的区分度，不能衡量其他分数。

2、Divergence

计算公式如下：

其中，u表示好、坏分数的均值，var表示好、坏分数的标准差。Divergence越大说明区分度越好。

3、Gini系数

GINI统计值衡量坏账户数在好账户数上的的累积分布与随机分布曲线之间的面积，好账户与坏账户分布之间的差异越大，GINI指标越高，表明模型的风险区分能力越强。

GINI系数的计算步骤如下：

（1）计算每个评分区间的好坏账户数。

（2） 计算每个评分区间的累计好账户数占总好账户数比率（累计good%）和累计坏账户数占总坏账户数比率(累计bad%)。

（3） 按照累计好账户占比和累计坏账户占比得出下图所示曲线ADC。

（4）计算出图中阴影部分面积，阴影面积占直角三角形ABC面积的百分比，即为GINI系数。



如上图Gini系数=

 

（二）模型区稳定度

群体稳定性指标PSI(Population Stability Index)是衡量模型的预测值与实际值偏差大小的指标。一般psi是在放款观察期（如6个月）后开始计算，来判断模型的稳定情况，如果出现比较大的偏差再进行模型的调整。说的明白些PSI表示的就是按分数分档后，针对不同样本，或者不同时间的样本，population分布是否有变化，就是看各个分数区间内人数占总人数的占比是否有显著变化，通常要求psi<0.25。公式如下：

PSI = sum（（实际占比-预期占比）* ln（实际占比/预期占比））



PSI实际应用范例：

（1）样本外测试，针对不同的样本测试一下模型稳定度，比如训练集与测试集，也能看出模型的训练情况，我理解是看出模型的方差情况。

（2）时间外测试，测试基准日与建模基准日相隔越远，测试样本的风险特征和建模样本的差异可能就越大，因此PSI值通常较高。至此也可以看出模型建的时间太长了，是不是需要重新用新样本建模了。

特别：模型调优

模型需要进行必要的调优，当遇到如下情形时：

（1）监控结果不满足要求，如连续3个月的KS低于30%，AUC低于70%，PSI高于25% ；

（2）产品发生变化 额度提高，周期提高，利率降低 ；

（3）人群发生变化 准入政策发生变化 ；

（4）其他宏观因素发生变化。



# week 14: [k-nearest neighbours, linear regression, and the naive Bayes classifier.](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%202%20Week%2014.ipynb)

[solution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%202%20Week%2014%20Answers.ipynb)

# supervised learning

## KNN分类算法

1. 概述
KNN 可以说是最简单的分类算法之一，同时，它也是最常用的分类算法之一。注意：KNN 算法是有监督学习中的分类算法，它看起来和另一个机器学习算法 K-means 有点像（K-means 是无监督学习算法），但却是有本质区别的。

2. 核心思想
KNN 的全称是 K Nearest Neighbors，意思是 K 个最近的邻居。从这个名字我们就能看出一些 KNN 算法的蛛丝马迹了。K 个最近邻居，毫无疑问，K 的取值肯定是至关重要的，那么最近的邻居又是怎么回事呢？其实，KNN 的原理就是当预测一个新的值 x 的时候，根据它距离最近的 K 个点是什么类别来判断 x 属于哪个类别。听起来有点绕，还是看看图吧。

图中绿色的点就是我们要预测的那个点，假设 K=3。那么 KNN 算法就会找到与它距离最近的三个点（这里用圆圈把它圈起来了），看看哪种类别多一些，比如这个例子中是蓝色三角形多一些，新来的绿色点就归类到蓝三角了。

但是，当 K=5 的时候，判定就变成不一样了。这次变成红圆多一些，所以新来的绿点被归类成红圆。从这个例子中，我们就能看得出 K 的取值是很重要的。
明白了大概原理后，我们就来说一说细节的东西吧，主要有两个，K 值的选取和点距离的计算。

2.1 距离计算
要度量空间中点距离的话，有好几种度量方式，比如常见的曼哈顿距离计算、欧式距离计算等等。不过通常 KNN 算法中使用的是欧式距离。这里只是简单说一下，拿二维平面为例，二维空间两个点的欧式距离计算公式如下：


这个高中应该就有接触到的了，其实就是计算（x1,y1）和（x2,y2）的距离。拓展到多维空间，则公式变成这样：


这样我们就明白了如何计算距离。KNN 算法最简单粗暴的就是将预测点与所有点距离进行计算，然后保存并排序，选出前面 K 个值看看哪些类别比较多。但其实也可以通过一些数据结构来辅助，比如最大堆，这里就不多做介绍，有兴趣可以百度最大堆相关数据结构的知识。

2.2 K值选择
通过上面那张图我们知道 K 的取值比较重要，那么该如何确定 K 取多少值好呢？答案是通过交叉验证（将样本数据按照一定比例，拆分出训练用的数据和验证用的数据，比如6：4拆分出部分训练数据和验证数据），从选取一个较小的 K 值开始，不断增加 K 的值，然后计算验证集合的方差，最终找到一个比较合适的 K 值。



通过交叉验证计算方差后你大致会得到下面这样的图：


这个图其实很好理解，当你增大 K 的时候，一般错误率会先降低，因为有周围更多的样本可以借鉴了，分类效果会变好。但注意，和 K-means 不一样，当 K 值更大的时候，错误率会更高。这也很好理解，比如说你一共就35个样本，当你 K 增大到30的时候，KNN 基本上就没意义了。



所以选择 K 点的时候可以选择一个较大的临界 K 点，当它继续增大或减小的时候，错误率都会上升，比如图中的 K=10。

3. 算法实现
3.1 Sklearn KNN参数概述
要使用 Sklearn KNN 算法进行分类，我们需要先了解 Sklearn KNN 算法的一些基本参数：

def KNeighborsClassifier(n_neighbors = 5,
                       weights='uniform',
                       algorithm = '',
                       leaf_size = '30',
                       p = 2,
                       metric = 'minkowski',
                       metric_params = None,
                       n_jobs = None
                       )
其中：
n_neighbors：这个值就是指 KNN 中的 “K”了。前面说到过，通过调整 K 值，算法会有不同的效果。
weights（权重）：最普遍的 KNN 算法无论距离如何，权重都一样，但有时候我们想搞点特殊化，比如距离更近的点让它更加重要。这时候就需要 weight 这个参数了，这个参数有三个可选参数的值，决定了如何分配权重。参数选项如下：
* ‘uniform’：不管远近权重都一样，就是最普通的 KNN 算法的形式。

* ‘distance’：权重和距离成反比，距离预测目标越近具有越高的权重。

* 自定义函数：自定义一个函数，根据输入的坐标值返回对应的权重，达到自

定义权重的目的。

algorithm：在 Sklearn 中，要构建 KNN 模型有三种构建方式：
1. 暴力法，就是直接计算距离存储比较的那种方式。

2. 使用 Kd 树构建 KNN 模型。

3. 使用球树构建。

其中暴力法适合数据较小的方式，否则效率会比较低。如果数据量比较大一般会选择用 Kd 树构建 KNN 模型，而当 Kd 树也比较慢的时候，则可以试试球树来构建 KNN。参数选项如下：

* ‘brute’ ：蛮力实现；

* ‘kd_tree’：KD 树实现 KNN；

* ‘ball_tree’：球树实现 KNN ；

* ‘auto’： 默认参数，自动选择合适的方法构建模型。

不过当数据较小或比较稀疏时，无论选择哪个最后都会使用 ‘brute’。

leaf_size：如果是选择蛮力实现，那么这个值是可以忽略的。当使用 Kd 树或球树，它就是停止建子树的叶子节点数量的阈值。默认30，但如果数据量增多这个参数需要增大，否则速度过慢不说，还容易过拟合。


p：和 metric 结合使用，当 metric 参数是 “minkowski” 的时候，p=1 为曼哈顿距离， p=2 为欧式距离。默认为p=2。


metric：指定距离度量方法，一般都是使用欧式距离。
* ‘euclidean’ ：欧式距离；

* ‘manhattan’：曼哈顿距离；

* ‘chebyshev’：切比雪夫距离；

* ‘minkowski’： 闵可夫斯基距离，默认参数。

n_jobs：指定多少个CPU进行运算，默认是-1，也就是全部都算。


3.2 KNN代码实例
KNN 算法算是机器学习里面最简单的算法之一了，我们来看 Sklearn 官方给出的例子是怎样使用KNN 的。

数据集使用的是著名的鸢尾花数据集，用 KNN 来对它做分类。我们先看看鸢尾花长的啥样：


上面这个就是鸢尾花了，这个鸢尾花数据集主要包含了鸢尾花的花萼长度、花萼宽度、花瓣长度、花瓣宽度4个属性（特征），以及鸢尾花卉属于『Setosa、Versicolour、Virginica』三个种类中的哪一类（这三种都长什么样我也不知道）。



在使用 KNN 算法之前，我们要先决定 K 的值是多少。要选出最优的 K 值，可以使用 Sklearn 中的交叉验证方法，代码如下：

from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target
k_range = range(1, 31)
k_error = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
运行后，我们可以得到下面这样的图：


有了这张图，我们就能明显看出 K 值取多少的时候误差最小，这里明显是 K=11 最好。当然在实际问题中，如果数据集比较大，那为减少训练时间，K 的取值范围可以缩小。

有了 K 值我们就能运行 KNN 算法了，具体代码如下：

import matplotlib.pyplot as plt
from numpy import *
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 11

 # 导入一些要玩的数据
iris = datasets.load_iris()
x = iris.data[:, :2]  # 我们只采用前两个feature,方便画图在二维平面显示
y = iris.target

h = .02  # 网格中的步长

 # 创建彩色的图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#weights是KNN模型中的一个参数，上述参数介绍中有介绍，这里绘制两种权重参数下KNN的效果图
for weights in ['uniform', 'distance']:
    # 创建了一个knn分类器的实例，并拟合数据
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)

    # 绘制决策边界，为此，我们将为每个分配一个颜色
    # 来绘制网格中的点 [x_min, x_max]x[y_min, y_max].
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # 将结果放入一个彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练点
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.show()
4. 算法特点
KNN是一种非参的、惰性的算法模型。什么是非参，什么是惰性呢？



非参的意思并不是说这个算法不需要参数，而是意味着这个模型不会对数据做出任何的假设，与之相对的是线性回归（我们总会假设线性回归是一条直线）。也就是说 KNN 建立的模型结构是根据数据来决定的，这也比较符合现实的情况，毕竟在现实中的情况往往与理论上的假设是不相符的。



惰性又是什么意思呢？想想看，同样是分类算法，逻辑回归需要先对数据进行大量训练（tranning），最后才会得到一个算法模型。而 KNN 算法却不需要，它没有明确的训练数据的过程，或者说这个过程很快。



5. 算法优缺点
5.1优点
简单易用。相比其他算法，KNN 算是比较简洁明了的算法，即使没有很高的数学基础也能搞清楚它的原理。
模型训练时间快，上面说到 KNN 算法是惰性的，这里也就不再过多讲述。
预测效果好。
对异常值不敏感。
5.2 缺点
对内存要求较高，因为该算法存储了所有训练数据。
预测阶段可能很慢。
对不相关的功能和数据规模敏感。
6. KNN 和 K-means比较
前面说到过，KNN 和 K-means 听起来有些像，但本质是有区别的，这里我们就顺便说一下两者的异同吧。

6.1 相同点：
K 值都是重点。
都需要计算平面中点的距离。
6.2 相异点：

KNN 和 K-means 的核心都是通过计算空间中点的距离来实现目的，只是他们的目的是不同的。KNN 的最终目的是分类，而 K-means 的目的是给所有距离相近的点分配一个类别，也就是聚类。

简单说，就是画一个圈，KNN 是让进来圈子里的人变成自己人，K-means 是让原本在圈内的人归成一类人。

至于什么时候应该选择使用 KNN 算法，Sklearn 的这张图给了我们一个答案：


简单来说，就是当需要使用分类算法，且数据比较大的时候就可以尝试使用 KNN 算法进行分类了。



补充：

Scikit-learn (Sklearn) 是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。当我们面临机器学习问题时，便可根据上图来选择相应的方法。Sklearn 具有以下特点：

简单高效的数据挖掘和数据分析工具；
让每个人能够在复杂环境中重复使用；
建立NumPy、Scipy、MatPlotLib 之上。

## 线性回归

[线性回归面试题](https://zhuanlan.zhihu.com/p/66519299)


线性回归算法原理与总结

什么是线性回归
和之前介绍的K近邻算法不同，K近邻主要是解决分类问题，而线性回归顾名思义是用来解决回归问题的。而线性回归具有如下特征：

解决回归问题
思想简单，实现容易
许多强大的非线性模型的基础，比如逻辑回归、多项式回归、svm等等
结果具有很好的可解释性
蕴含机器学习中的很多重要思想
那么什么是线性回归呢？我们画一张图：

![image](https://user-images.githubusercontent.com/77952995/155894339-dc9b6470-8c47-4e46-8a35-75e6a28525be.png)

图中是房屋的面积与价格之间的对应关系，不同的面积对应不同的价格，由此在二维平面中便形成了多个点。我们的目的就是要找到一条直线，最大程度上来拟合这些点。

但是在之前的K近邻中，横轴和纵轴都是样本的特征，而标签则是由这个点是红色还是蓝色决定的。但是在线性回归中，由于是房产数据，我们必须要预测出一个具体的数值，而不能像分类问题那样，用简单的颜色来代表类别。而这些数据显然是在一个连续的样本空间中，因此需要一个坐标轴来表示。也正因为如此，在二维平面中只能有一个特征，要是多个特征，我们就要在更高的维度上进行观察了。

如果样本的特征只有一个，我们称之为简单线性回归。

我们的目的是要找到一个直线来尽可能多的拟合这些点，而在二维平面上显然是 𝑦=𝑎𝑥+𝑏 ，那么对于每一个样本 𝑥，都会有一个真实值 𝑦 和使用拟合曲线预测出来的预测值 𝑦̂  ，因此我们的真实值和预测值就会有一个差距。而 𝑦=𝑎𝑥+𝑏 可以对应平面中的任何曲线，我们的目的显然是找到真实值和预测值之间的差距能达到最小的那根曲线。

因此如果使用数学来描述的话就是：对于每一个样本点 𝑥𝑖 ，都有一个真实值 𝑦𝑖 ，和一个预测值 𝑦̂ 𝑖=𝑎𝑥𝑖+𝑏 ，而我们希望 𝑦𝑖 和 𝑦̂ 𝑖 之间的差距尽量小。如果考虑所有样本的话，那么就是所有样本的预测值和真实值之差的平方和 ∑𝑖=1𝑚(𝑦𝑖−𝑦̂ 𝑖)2 最小。

既然有真实值和预测值，那么评价一个直线的拟合程度，就看所有样本的真实值和预测值之差。但是显然我们不能直接将两者之差加在一起，那么两者之差可能有正有负，会抵消掉；若是取绝对值的话，那么我们知道，这不是一个处处可导的函数，因为我们后面在求系数和截距的时候，是需要函数处处可导的。那么显然容易想到取两者之差的平方，而且也将正负的问题解决了，然后再将m个样本进行相加即可。

因此我们最终的目的就是找到一个合适的 𝑎 和 𝑏 ，使得 ∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)2 达到最小，然后通过 𝑎 和 𝑏 来确定我们的拟合曲线。

我们可以对式子进行化简，但是我们先不着急这么做，我们可以先看看机器学习的思路。

我们的目标是找到 𝑎 和 𝑏 ，使得 ∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)2 达到最小，而 ∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)2 便被称为"损失函数(loss function)"，与之相对的还有一个"效用函数(utility function)"。

近乎所有的参数学习算法，都是这个模式。得出一个损失函数或者一个效用函数，两者统称为目标函数。通过不断地优化，使得损失函数的值达到最小，或者效用函数的值达到最大。这不是特定的某个算法，而是近乎所有的参数学习都是这个套路，比如：线性回归、多项式回归、逻辑回归、SVM、神经网络等等。本质上，都是在学习参数，来最优化目标函数。只不过由于模型的不同，因此建立的参数、优化的方式也不同。

也正因为机器学习的大部分算法都拥有这个特征，于是有了一门学科，叫最优化原理。

回到我们的问题，那么我们如何才能求出 𝑎 和 𝑏 呢？显然要通过求导的方式，如果你还记得高中学习过的数学知识，那么你会发现这不就是最小二乘法嘛。下面我们就来推导一下，当然啦，如果不感兴趣可以只看结论，但是最好还是要掌握一下推导过程，对自身是有帮助的。

我们将 ∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)2 记作为 𝐹，那么有 𝐹=∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)2 ，而我们的目的显然是对 𝑎 和 𝑏 求偏导。

我们先对 𝑏 求偏导，过程如下：

𝛿𝐹𝛿𝑏=∑𝑖=1𝑚2(𝑦𝑖−𝑎𝑥𝑖−𝑏)∗−1

𝛿𝐹𝛿𝑏=−2∑𝑖=1𝑚(𝑦𝑖−𝑎𝑥𝑖−𝑏)

𝛿𝐹𝛿𝑏=−2(∑𝑖=1𝑚𝑦𝑖−𝑎∑𝑖=1𝑚𝑥𝑖−∑𝑖=1𝑚𝑏)

𝛿𝐹𝛿𝑏=−2(𝑚𝑦⎯⎯⎯−𝑎𝑚𝑥⎯⎯⎯−𝑚𝑏)

我们下面再对 𝑎 求偏导，过程如下：

𝛿𝐹𝛿𝑎=∑𝑖=1𝑚2(𝑦𝑖−𝑎𝑥𝑖−𝑏)∗−𝑥𝑖

𝛿𝐹𝛿𝑎=−2∑𝑖=1𝑚(𝑥𝑖𝑦𝑖−𝑎(𝑥𝑖)2−𝑏𝑥𝑖)

𝛿𝐹𝛿𝑎=−2(∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑎∑𝑖=1𝑚(𝑥𝑖)2−𝑏∑𝑖=1𝑚𝑥𝑖)

然后令 𝛿𝐹𝛿𝑏=−2(𝑚𝑦⎯⎯⎯−𝑎𝑚𝑥⎯⎯⎯−𝑚𝑏)=0，得到 𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯−𝑏=0，解得 𝑏=𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯

此时 𝑏 我们就求了出来，然后再令 𝛿𝐹𝛿𝑎=−2(∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑎∑𝑖=1𝑚(𝑥𝑖)2−𝑏∑𝑖=1𝑚𝑥𝑖)=0，得到 ∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑎∑𝑖=1𝑚(𝑥𝑖)2−𝑚𝑏𝑥⎯⎯⎯=0

然后将 𝑏=𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯ 带进去，得到 ∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑎∑𝑖=1𝑚(𝑥𝑖)2−(𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯)𝑚𝑥⎯⎯⎯=0

∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑎∑𝑖=1𝑚(𝑥𝑖)2−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯+𝑎𝑚𝑥⎯⎯⎯2=0

∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯=𝑎(∑𝑖=1𝑚(𝑥𝑖)2−𝑚𝑥⎯⎯⎯2)

𝑎=∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯∑𝑖=1𝑚(𝑥𝑖)2−𝑚𝑥⎯⎯⎯2

又因为 ∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)(𝑦𝑖−𝑦⎯⎯⎯)=∑𝑖=1𝑚(𝑥𝑖𝑦𝑖−𝑥𝑖𝑦⎯⎯⎯−𝑥⎯⎯⎯𝑦𝑖+𝑥⎯⎯⎯𝑦⎯⎯⎯)=∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯+𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯=∑𝑖=1𝑚𝑥𝑖𝑦𝑖−𝑚𝑥⎯⎯⎯𝑦⎯⎯⎯

又因为 ∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)2=∑𝑖=1𝑚((𝑥𝑖)2−2𝑥⎯⎯⎯𝑥𝑖+𝑥⎯⎯⎯2)=∑𝑖=1𝑚(𝑥𝑖)2−2𝑚𝑥⎯⎯⎯2+𝑚𝑥⎯⎯⎯2=∑𝑖=1𝑚(𝑥𝑖)2−𝑚𝑥⎯⎯⎯2

所以 𝑎=∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)(𝑦𝑖−𝑦⎯⎯⎯)∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)2

到此，我们便找到了最理想的 𝑎 和 𝑏 ，其中 𝑎=∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)(𝑦𝑖−𝑦⎯⎯⎯)∑𝑖=1𝑚(𝑥𝑖−𝑥⎯⎯⎯)2， 𝑏=𝑦⎯⎯⎯−𝑎𝑥⎯⎯⎯

整个过程还是比较简单的，如果你的数学基础还没有忘记的话，主要是最后的那两步替换比较巧妙。

当然这个式子也不用刻意去记，网上一大堆，需要的时候去查找就是了，重要的是理解整个推导过程。

我们上面推导的是简单线性回归，也就是要求样本只有一个特征，但是显然真实的样本，特征肯定不止一个。如果预测的样本有多个特征的话，那么便称之为多元线性回归，我们后面会介绍。

简单线性回归的实现
下面我们就使用代码来实现简单线性回归，先来看一下简单的数据集。

import numpy as np
import plotly.graph_objs as go

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 5])

trace0 = go.Scatter(x=x, y=y, mode="markers", marker={"size": 10})
figure = go.Figure(data=[trace0], layout={"showlegend": False, "template": "plotly_dark"})
figure.show()


可以看到数据绘制出来，大致长上面这样，下面就来按照公式推导我们的 𝑎 和 𝑏 。

a = np.sum( (x - np.mean(x)) * (y - np.mean(y)) ) / np.sum( (x - np.mean(x)) ** 2 )
b = np.mean(y) - a * np.mean(x)
print(a, b)  # 0.8 0.39999999999999947
此时 𝑎 和 𝑏 便被计算了出来，那么我们就将这条直线给绘制出来吧。

y_hat = a * x + b

trace0 = go.Scatter(x=x, y=y, mode="markers", marker={"size": 10})
trace1 = go.Scatter(x=x, y=y_hat, mode="lines", marker={"size": 10})
figure = go.Figure(data=[trace0, trace1], layout={"showlegend": False, "template": "plotly_dark"})
figure.show()


然后我们进行预测。

x_predict = 6
y_predict = a * x_predict + b
print(y_predict)  # 5.2
肿么样，似不似灰常简单呢？那么下面我们就按照sklearn的模式，来封装这个简单线性回归算法。

import numpy as np


class SimpleLinearRegression:

    def __init__(self):
        """
        关于变量后面加上一个下划线这种命名方式在sklearn中是有约定的
        表示这种变量不是由用户传来的，而是一开始就有的，具体是什么值则是由用户传来的样本进行训练之后得到的
        并且还可以提供给用户使用，这种类型的变量都会在后面加上一个下划线
        """
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """通过x_train和y_train训练模型"""
        assert x_train.ndim == 1, "简单线性回归只支持一个具有一个特征的训练集"
        assert len(x_train) == len(y_train), "样本数必须和标签数保持一致"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean

        # 按照sklearn的标准，我们要将这个实例进行返回
        return self

    def predict(self, x_predict):
        """对传来的x进行预测"""
        assert self.a_ is not None, "预测(predict)之前要先拟合(fit)"
        if isinstance(x_predict, list):
            x_predict = np.array(x_predict)
        return self.a_ * x_predict + self.b_

    def __str__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"

    def __repr__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"


x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 5])
sim_linear_reg = SimpleLinearRegression()
sim_linear_reg.fit(x, y)
print(sim_linear_reg.a_)  # 0.8
print(sim_linear_reg.b_)  # 0.39999999999999947

x_predict = [1, 3, 4, 6]
print(sim_linear_reg.predict(x_predict))  # [1.2 2.8 3.6 5.2]
可以看到，计算的结果是一致的。此时我们便通过sklearn的方式实现了一个简单线性回归

衡量线性回归的指标
在knn中，我们将数据集分为训练集合测试集，训练集用于训练模型，然后对测试集进行预测，通过预测出来的标签和真实标签进行对比，得到一个分类的准确度。

但是对于回归算法来说，我们如何评价一个算法呢？其实对于回归算法来说，我们仍然需要将数据集分为训练集和测试集，而我们的目的就是找到一个a和b，使得在训练集中，预测值和真实值之间差距尽可能的小。

所以我们的目标就变成了，找到 𝑎 和 𝑏 ，使得 ∑𝑖=1𝑚(𝑦𝑖𝑡𝑟𝑎𝑖𝑛−𝑎𝑥𝑖𝑡𝑟𝑎𝑖𝑛−𝑏)2 尽可能的小；然后对预测集进行预测，得到的误差 ∑𝑖=1𝑚(𝑦𝑖𝑡𝑒𝑠𝑡−𝑎𝑥𝑖𝑡𝑒𝑠𝑡−𝑏)2 便是我们的评价指标，越小代表模型越好。

但是这样又出现了一个问题，那就是我们的预测结果是和样本个数m有关的。如果A预测10000个样本，误差是1000，而B预测10个样本误差就达到800了，难道说B的算法比A好吗？显然是不能的，那么我们可以改进一下。

那么下面就产生了三个指标：

1. 均方误差(Mean Squared Error，MSE)

1𝑚∑𝑖=1𝑚(𝑦𝑖𝑡𝑒𝑠𝑡−𝑦̂ 𝑖𝑡𝑒𝑠𝑡)2

让衡量标准与我们的样本数无关，这便是均方误差。但是呢？这个均方误差还有一个潜在的问题，那就是如果我们预测房产数据单位是万元，那么得到的单位就是万元的平方，会造成量纲上的问题。 由此产生了下面一个衡量标准

2. 均方根误差(Root Mean Squared Error，RMSE)

1𝑚∑𝑖=1𝑚(𝑦𝑖𝑡𝑒𝑠𝑡−𝑦̂ 𝑖𝑡𝑒𝑠𝑡)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√=𝑀𝑆𝐸𝑡𝑒𝑠𝑡‾‾‾‾‾‾‾‾√

可以看到，我们直接对均方误差进行开根，便得到了均方根误差，这样就避免了量纲的问题。其实这两个衡量标准本质上比较类似，因为均方误差大，那么均方根误差也会跟着大，只是均方根误差更具有统计学上的意义

3. 平均绝对误差(Mean Absolute Error，MAE)

1𝑚∑𝑖=1𝑚|𝑦𝑖𝑡𝑒𝑠𝑡−𝑦̂ 𝑖𝑡𝑒𝑠𝑡|

这个衡量标准非常的直白，就是每一个样本的真实值和预测值之差的绝对值之和，这便是平均绝对误差。之前我们说过，在训练的时候，不建议使用这个模型，因为这不是一个处处可导的函数，但是在预测的时候，我们是完全可以使用这个指标的。也就是说，我们在评价一个算法所使用的指标和训练模型所使用的目标函数是可以不一致的。

使用程序来实现三个指标
这次我们使用sklearn中的真实数据集，波士顿房产数据。但由于我们这里还是简单线性回归，而波士顿房产数据有多个特征，因此我们这里只选择一个特征，选择房子的房间数。

from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data[:, 5]  # 选择房屋的房间数
y = boston.target
trace0 = go.Scatter(x=x, y=y, mode="markers", marker={"size": 10})
figure = go.Figure(data=[trace0], layout={"showlegend": False, "template": "plotly_dark"})
figure.show()


数据集大致长这样，但是会发现有些奇怪的点，对，就是纵轴为50的位置。之所以出现这种情况，是因为在实际生活中，没有进行更仔细的分类，比如价格超过50万美元，就按照50万美元来统计了。比如我们经常看到的统计月薪，最下面的选项是2000及以下，最上面的是40000及以上等等。 因此我们可以将最大值给去掉：

x = boston.data[:, 5] 
y = boston.target
x = x[y < np.max(y)]
y = y[y < np.max(y)]

trace0 = go.Scatter(x=x, y=y, mode="markers", marker={"size": 10})
figure = go.Figure(data=[trace0], layout={"showlegend": False, "template": "plotly_dark"})
figure.show()


如此一来，我们便把最大值给去掉了。

from sklearn.model_selection import train_test_split


class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        assert x_train.ndim == 1, "简单线性回归只支持一个具有一个特征的训练集"
        assert len(x_train) == len(y_train), "样本数必须和标签数保持一致"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert self.a_ is not None, "预测(predict)之前要先拟合(fit)"
        if isinstance(x_predict, list):
            x_predict = np.array(x_predict)
        return self.a_ * x_predict + self.b_

    def __str__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"

    def __repr__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"


boston = load_boston()
x = boston.data[:, 5]  # 选择房屋的房间数
y = boston.target

x = x[y < np.max(y)]
y = y[y < np.max(y)]

x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_train.shape)  # (367,)
print(x_test.shape)  # (123,)

linear = SimpleLinearRegression()
linear.fit(x_train, y_train)

print(linear.a_)  # 7.920613164194192
print(linear.b_)  # -27.860720801665288
我们将得到的直线绘制出来：

trace0 = go.Scatter(x=x_train, y=y_train, mode="markers", marker={"size": 10})
trace1 = go.Scatter(x=x_train, y=linear.a_ * x_train + linear.b_, mode="lines", marker={"size": 10})
figure = go.Figure(data=[trace0, trace1], layout={"showlegend": False, "template": "plotly_dark"})
figure.show()


然后我们使用上面的三个标准来计算它们的误差：

y_predict = linear.predict(x_test)
### MSE
mse = np.sum((y_predict - y_test) ** 2) / len(y_test)
### RMSE
rmse = np.sqrt(mse)
### MAE
mae = np.sum(np.abs(y_predict - y_test)) / len(y_test)
print(mse)  # 25.9281717303928
print(rmse)  # 5.091971301018183
print(mae)  # 3.795667974965169
可以看出还是很简单的，然后老规矩，下面肯定要看看sklearn中，这些衡量标准。首先sklearn没有RMSE，我们可以先求出MSE，再手动开根。

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print(mean_absolute_error(y_test, y_predict))  # 25.9281717303928
print(np.sqrt(mean_squared_error(y_test, y_predict)))  # 5.091971301018183
print(mean_squared_error(y_test, y_predict))  # 3.795667974965169
关于RMSE和MAE，我们知道这两种衡量标准它们在量纲上是一致的，但是哪个更好呢？根据我们上面预测的结果发现RMSE大于MAE，没错，我们观察一下式子。因为RMSE进行了平方，虽然又进行了开根，但是整体来说还是有将错误放大的趋势，而MAE没有，因此我们将RMSE作为评价指标会更有意义。

介绍了这么多指标，你以为就完了，其实我们还有一个最好的衡量线性回归法的指标，在sklearn的线性回归中，写入score函数中的也是这个指标。

最好的衡量线性回归法的指标：R Squared
事实上，无论是RMSE还是MAE，是随着样本的不同而变化的。比如房产数据，可能预测出来的误差是五万美元，学生成绩预测出来的误差是10分，那么我们的算法是作用在房产数据好呢？还是作用在学生成绩好呢？这些是没有办法比较的。而分类问题不同，分类是根据样本的真实特征和预测特征进行比较，看看预测对了多少个，然后除以总个数，得到的准确度是在0到1之间的。越接近1越好，越接近0越差，而RMSE和MAE由于样本种类的不同，导致是无法比较的。

于是，我们便有了下一个指标R Squared，也就是所谓的 𝑅2 。

𝑅2=1−∑(𝑦̂ 𝑖−𝑦𝑖)2∑(𝑦⎯⎯⎯𝑖−𝑦𝑖)2=使用我们的模型预测时产生的错误使用𝑦=𝑦⎯⎯⎯预测时产生的错误

怎么理解这个式子呢？首先把 𝑦 的平均值看成一个模型，这个模型也叫作基准模型，分母就是用基准模型预测出来的误差。换句话说，这个结果是与样本特征是无关的，因为它没有用到 𝑥 ，那么这个基准模型必然会产生很多错误。而分子则是使用我们训练出来模型进行预测出来误差，如果我们训练出来的模型预测时产生的错误除以基准模型预测时产生的错误的结果越小，那么再用1去减，得到的结果就越大，说明我们训练的模型就越好。

根据 𝑅2 我们可以得出如下结论：

𝑅2 <= 1
𝑅2 越大越好，当我们的模型不犯任何错误时，𝑅2 取得最大值1
当我们的模型等于基准模型(Baseline Model)时，𝑅2 为0
如果 𝑅2 < 0，说明我们学习到的模型还不如基准模型。那么很有可能我们的数据压根就不存在所谓的线性关系，不管是正相关还是负相关。
我们再来看看这个 𝑅2 ，如果我们对式子进行一下变换的话。

𝑅2=1−∑(𝑦̂ 𝑖−𝑦𝑖)2∑(𝑦⎯⎯⎯𝑖−𝑦𝑖)2=1−∑(𝑦̂ 𝑖−𝑦𝑖)2/𝑚∑(𝑦⎯⎯⎯𝑖−𝑦𝑖)2/𝑚=1−𝑀𝑆𝐸(𝑦̂ ,𝑦)𝑉𝑎𝑟(𝑦)

会发现这个 𝑅2 就等于 1 - MSE/Var(y)

在sklearn中也提供了R Squared。

from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))  # 0.5107683519296282
### 或者手动计算也可以
mse = mean_squared_error(y_test, y_predict)
r2_score = 1 - mse / np.var(y_test)
print(r2_score)  # 0.5107683519296282
我们在K近邻中，会直接封装一个score函数，返回分类的准确度，那么在sklearn中，我们也依旧可以封装一个score。另外在sklearn的线性回归中，score函数也是直接返回了R Squared，而且sklearn的线性回归是支持多元线性回归的，后面会说。因此如果只想看分类的准确度可以调用score方法，如果想要预测出来的值，那么可以调用predict方法。

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        assert x_train.ndim == 1, "简单线性回归只支持一个具有一个特征的训练集"
        assert len(x_train) == len(y_train), "样本数必须和标签数保持一致"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert self.a_ is not None, "预测(predict)之前要先拟合(fit)"
        if isinstance(x_predict, list):
            x_predict = np.array(x_predict)
        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        mse = mean_squared_error(y_test, y_predict)
        r2_score = 1 - mse / np.var(y_test)
        return r2_score

    def __str__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"

    def __repr__(self):
        return f"<SimpleLinearRegression>:a_={self.a_},b_={self.b_}"


boston = load_boston()
x = boston.data[:, 5]  # 选择房屋的房间数
y = boston.target

x = x[y < np.max(y)]
y = y[y < np.max(y)]

x_train, x_test, y_train, y_test = train_test_split(x, y)

linear = SimpleLinearRegression()
linear.fit(x_train, y_train)

print(linear.score(x_test, y_test))  # 0.5849894775958314
多元线性回归和正规方程解
我们之前一直解决的是简单线性回归，也就是我们的样本只有一个特征。但是实际中，样本是有很多很多特征的，成百上千个也是很正常的。对于有多个特征的样本，我们依旧可以使用线性回归法来解决，我们称之为多元线性回归。



还是以这张图为例，在简单线性回归中，我们的每一个x只是一个值。但是现在在多元线性回归中，每个x则是一个数组(或者说是向量)，具有n个特征。那么我们的方程就变成了 𝑦=θ0+θ1𝑥1+θ2𝑥2+...+θ𝑛𝑥𝑛 ，n个特征对应的系数则是 θ0,θ1,θ2,......,θ𝑛，而且此时仍然是有一个截距的，我们记作 θ0 ，所以在预测模型的时候，也是有n个系数。

可以看到多元线性回归和简单线性回归并无本质上的不同，只不过在简单线性回归中，我们只需要求一个截距和一个系数即可，也就是 θ0 和 θ1；但是在多元线性回归中，我们将只求一个截距和一个系数扩展到了求一个截距和n个系数。

我们在简单线性回归中，目的是使得：∑𝑖=1𝑚(𝑦𝑖−𝑦̂ 𝑖)2 尽可能的小，此时在多元线性回归中，我们的目的依然是使得这个式子尽可能的小，因为它表达的便是真实值和预测值之间误差。只不过此时的 𝑦̂ 𝑖=θ0+θ1𝑋𝑖1+θ2𝑋𝑖2+......+θ𝑛𝑋𝑖𝑛 ，其中 𝑥𝑖=(𝑋𝑖1,𝑋𝑖2,......,𝑋𝑖𝑛)，而我们是要找到一组 θ0,θ1,θ2,......,θ𝑛 ，来完成这一点。

𝑦̂ 𝑖=θ0+θ1𝑋𝑖1+θ2𝑋𝑖2+......+θ𝑛𝑋𝑖𝑛，θ=(θ0,θ1,θ2,...,θ𝑛)𝑇

𝑦̂ 𝑖=θ0𝑋𝑖0+θ1𝑋𝑖1+θ2𝑋𝑖2+......+θ𝑛𝑋𝑖𝑛，𝑋𝑖0≡1，𝑋𝑖=(𝑋𝑖0,𝑋𝑖1,𝑋𝑖2,...,𝑋𝑖𝑛)

𝑦̂ 𝑖=𝑋𝑖·θ

我们找到一个 θ，让 θ=(θ0,θ1,θ2,...,θ𝑛)𝑇  ，注意这里的 θ 是一个列向量，为了方便我们写成行向量的模式，然后再进行转置。但是我们发现在模型中，对于 θ0 ，没有 𝑋 与之配对，怪孤单的，因为我们可以虚构出一个特征 𝑋0 ，让其与 θ0 结合，这样模型中的每一个部分都满足相同的规律。但是由于 𝑋0 这个特征使我们虚构出来的，因此这个 𝑋0 是恒等于1的。

我们可以对式子再进行一下推广：

𝑋𝑏=⎡⎣⎢⎢⎢⎢11...1𝑋11𝑋21𝑋𝑚1𝑋12𝑋22𝑋𝑚2𝑋13𝑋23𝑋𝑚3.........𝑋1𝑛𝑋2𝑛...𝑋𝑚𝑛⎤⎦⎥⎥⎥⎥

θ=⎡⎣⎢⎢⎢⎢⎢θ0θ1θ2...θ𝑛⎤⎦⎥⎥⎥⎥⎥

由此可以得出：

𝑦̂ =𝑋𝑏·θ

所以我们的目标就变成了使 (𝑦−𝑋𝑏·θ)𝑇(𝑦−𝑋𝑏·θ) 尽可能小。

这个式子很简单，关于为什么第一个 (𝑦−𝑋𝑏·θ) 要进行转置，如果不进行转置的话，那么 (𝑦−𝑋𝑏·θ)(𝑦−𝑋𝑏·θ) 得到的还是一个向量。我们的目标函数是对每一个样本的真实值和预测值之间的误差进行了求和，那么我们如果不转置的话，还要进行一个求和运算，如果对第一项进行转置的话，那么点乘之后会自动进行相加，因此这种表示方法更为直观。

因此我们的目标函数就是找到一个θ，使得目标函数尽可能的小。这里推导过程就不给出了，就是下面的θ，有兴趣的话可以去网上搜索一下推导过程。

θ=(𝑋𝑇𝑏𝑋𝑏)−1𝑋𝑇𝑏𝑦

此时的这个θ，我们就称之为正规方程解。关于这个式子也没有必要背，因为在网上一搜就是，其实在真正使用多元线性回归的时候，我们还有另外的方法可以估计出这个θ的值，这个另外的方法，我们之后介绍。

另外我们在求解这个θ的时候，实际上也会有一些问题的，那就是时间复杂度高，为O(n3)，优化之后也有O(n2.4)。但是好处就是不需要对数据进行归一化处理，因为我们就是对原始的数据进行运算，这种运算是不存在量纲的问题的。

实现多元线性回归
那么我们就来实现一下多元线性回归。

```python
import numpy as np


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        # 我们需要计算θ，但是呢，这个θ是不需要用户访问，因此我们设置为被保护的
        self._theta = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "样本的个数和标签的个数要保持一致"
        # 计算Xb,这个Xb就是样本的基础上再加上一列
        X_b = np.c_[np.ones(len(X_train)), X_train]

        self._theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.coef_ is not None, "预测(predict)之前必须先fit(拟合)"
        assert X_predict.shape[1] == len(self.coef_), "样本的特征数要和相应的系数保持一致"

        X_b = np.c_[np.ones(len(X_predict)), X_predict]

        predict = X_b @ self._theta
        return predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)

        mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
        var = np.var(y_test)
        r2_score = 1 - mse / var
        return r2_score


if __name__ == '__main__':
    from sklearn.datasets import load_boston

    boston = load_boston()
    # 选择所有的特征
    X = boston.data
    y = boston.target

    index = np.arange(0, X.shape[0])
    test_size = 0.3
    np.random.shuffle(index)

    X_train, X_test, y_train, y_test = (X[index[int(X.shape[0] * test_size):]], X[index[: int(X.shape[0] * test_size)]],
                                        y[index[int(X.shape[0] * test_size):]], y[index[: int(X.shape[0] * test_size)]])

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    print(linear.coef_)
    """
    [-1.26398128e-01  3.57329311e-02 -9.99460755e-03  2.49336361e+00
     -1.66228834e+01  3.58359972e+00 -1.04839104e-03 -1.39258694e+00
      3.43283440e-01 -1.30859190e-02 -1.00377059e+00  1.06387337e-02
     -5.33110952e-01]
    """
    print(linear.interception_)  # 38.10416228778042
    print(linear.score(X_test, y_test))  # 0.7907505239934073 
    
   ```
可以看到当我们使用所有的特征时，结果的R Squared是比我们之前只使用一个特征要好。

那么最后我们还是要来看看sklearn中给我们提供的线性回归算法。

   ```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
linear = LinearRegression()
### 可以看到都是这种模式，我们不需要管里面的逻辑是如何实现的
### 只需要将训练集的特征和标签传进去进行训练即可
linear.fit(X_train, y_train)
### 得到训练的模型之后，可以调用predict，传入X_test得到预测的结果
### 如果只想看分类的准确度，也可以调用score，直接传入X_test和y_test得到准确度
print(linear.score(X_test, y_test))  # 0.7712210030661906
   ```
   
之前我们介绍过knn，实际上Knn不仅可以解决分类问题，还可以解决回归问题，我们来看看sklearn中如何使用Knn进行回归。
   ```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
linear = KNeighborsRegressor()
linear.fit(X_train, y_train)
print(linear.score(X_test, y_test))  # 0.5733855243681796
   ```
可以看到，效果没有线性回归好。但是呢？我们在初始化的时候，没有指定超参数，那么我们需要进行网格搜索，来找到一个最好的超参数，看看相应的score是多少。

   ```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
linear = KNeighborsRegressor()
grid_params = [
    {
        "n_neighbors": list(range(1, 10)),
        "weights": ["uniform"]
    },
    {
        "n_neighbors": list(range(1, 10)),
        "weights": ["distance"],
        "p": list(range(1, 6))
    }
]

cv = GridSearchCV(linear, grid_params, n_jobs=-1)
cv.fit(X_train, y_train)
best_params = cv.best_params_
best_estimator = cv.best_estimator_

print(best_params)  # {'n_neighbors': 4, 'p': 1, 'weights': 'distance'}
print(best_estimator.score(X_test, y_test))  # 0.7032252003791625
"""
结果显示，当n_neighbors=4，p=1也就是去曼哈顿距离，weights按照距离反比的时候，效果最好
分数为0.7032252003791625，比默认的要好不少，但分数依旧是没有线性回归高的
"""
   ```
## 逻辑回归

[面试问题](https://zhuanlan.zhihu.com/p/46591702)

逻辑回归模型是针对线性可分问题的一种易于实现而且性能优异的分类模型。
它假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降法来求解参数，来达到将数据二分类的目的。

算法推导
引入几率比（odds）：指一个事件发生的概率与不发生概率的比值。对其求log，可得：
𝑙𝑜𝑔𝑖𝑡(𝑝)=log𝑝1−𝑝
将对数几率记为输入特征值的线性表达式,可得

𝑙𝑜𝑔𝑖𝑡(𝑃(𝑌=1|𝑋))=𝑤𝑇𝑥
对于某一样本属于特定类别的概率，为𝑙𝑜𝑔𝑖𝑡函数的反函数，称为𝑙𝑜𝑔𝑖𝑠𝑡𝑖𝑐函数，即𝑠𝑖𝑔𝑚𝑜𝑖𝑑函数:

𝜙(𝑥)=11+𝑒−𝑧
逻辑斯蒂回归采用sigmoid函数作为激励函数

逻辑斯蒂回归模型定义：
𝑃(𝑌=1|𝑋)=ℎ𝜃(𝑥)
𝑃(𝑌=0|𝑋)=1−ℎ𝜃(𝑥)
可知，输出𝑌=1的对数几率是输入𝑥的线性函数。

对于给定的训练数据集𝑇,可以应用极大似然估计法估计模型参数，假设模型概率分布是：
𝑃(𝑌=1|𝑋)=ℎ𝜃(𝑥)
𝑃(𝑌=0|𝑋)=1−ℎ𝜃(𝑥)
似然函数为：

∏𝑖=1𝑁[ℎ𝜃(𝑥𝑖)]𝑦𝑖[1−ℎ𝜃(𝑥𝑖)]1−𝑦𝑖
对数似然函数为：

𝑙(𝜃)=∑𝑖=1𝑁[𝑦𝑖logℎ𝜃(𝑥𝑖)+(1−𝑦𝑖)log(1−ℎ𝜃(𝑥𝑖))]
公式推导
我们使用梯度下降的思想来求解此问题，变换的表达式如下：

𝐽(𝜃)=−1𝑚𝑙(𝜃)
因为我们要使用当前的𝜃值通过更新得到新的𝜃值，所以我们需要知道𝜃更新的方向(即当前𝜃是加上一个数还是减去一个数离最终结果近)，所以得到𝐽(𝜃)后对其求导便可得到更新方向，求导过程如下：

∂𝐽(𝜃)∂𝜃𝑗=−1𝑚∑𝑖=1𝑚[(𝑦𝑖ℎ𝜃(𝑥𝑖)−1−𝑦𝑖1−ℎ𝜃(𝑥𝑖))∗∂ℎ𝜃(𝑥𝑖)∂𝜃𝑗]=−1𝑚∑𝑖=1𝑚[(𝑦𝑖ℎ𝜃(𝑥𝑖)−1−𝑦𝑖1−ℎ𝜃(𝑥𝑖))∗ℎ𝜃(𝑥𝑖)∗(1−ℎ𝜃(𝑥𝑖))∗𝑥𝑗𝑖]=1𝑚∑𝑖=1𝑚(ℎ𝜃(𝑥𝑖)−𝑦𝑖)𝑥𝑗𝑖
得到更新方向后便可使用下面的式子不断迭代更新得到最终结果:

𝜃𝑗:=𝜃𝑗−𝛼1𝑚∑𝑖=1𝑚(ℎ𝜃(𝑥𝑖)−𝑦𝑖)𝑥𝑗𝑖
优缺点
逻辑斯蒂回归模型的优点有：

形式简单，模型的可解释性非常好。从特征的权重可以看到不同的特征对最后结果的影响，某个特征的权重值比较高，那么这个特征最后对结果的影响会比较大。
模型效果不错。在工程上是可以接受的（作为baseline)，如果特征工程做的好，效果不会太差，并且特征工程可以大家并行开发，大大加快开发的速度。
训练速度较快。分类的时候，计算量仅仅只和特征的数目相关。并且逻辑回归的分布式优化𝑠𝑔𝑑发展比较成熟，训练的速度可以通过堆机器进一步提高，这样我们可以在短时间内迭代好几个版本的模型。
资源占用小,尤其是内存。因为只需要存储各个维度的特征值，。
方便输出结果调整。逻辑回归可以很方便的得到最后的分类结果，因为输出的是每个样本的概率分数，我们可以很容易的对这些概率分数进行cutoff，也就是划分阈值(大于某个阈值的是一类，小于某个阈值的是一类)。
逻辑斯蒂回归模型的缺点有：

准确率并不是很高。因为形式非常的简单(非常类似线性模型)，很难去拟合数据的真实分布。
很难处理数据不平衡的问题。举个例子：如果我们对于一个正负样本非常不平衡的问题比如正负样本比 10000:1.我们把所有样本都预测为正也能使损失函数的值比较小。但是作为一个分类器，它对正负样本的区分能力不会很好。
处理非线性数据较麻烦。逻辑回归在不引入其他方法的情况下，只能处理线性可分的数据，或者进一步说，处理二分类的问题
逻辑回归本身无法筛选特征。有时候，我们会用GBDT来筛选特征，然后再上逻辑回归。
相关问题
逻辑回归与线性回归区别？

本质上来说，两者都属于广义线性模型，但他们两个要解决的问题不一样，逻辑回归解决的是分类问题，输出的是离散值，线性回归解决的是回归问题，输出的连续值。另外，损失函数方面：线性模型是平方损失函数，而逻辑回归则是似然函数。

LR的损失函数为什么要使用极大似然函数作为损失函数？

将极大似然函数取对数以后等同于对数损失函数。在逻辑回归这个模型下，对数损失函数的训练求解参数的速度是比较快的。
梯度更新速度只和𝑥𝑖𝑗，𝑦𝑖相关。和𝑠𝑖𝑔𝑚𝑜𝑑函数本身的梯度是无关的。这样更新的速度是可以自始至终都比较的稳定。
为什么不选平方损失函数呢？其一是因为如果你使用平方损失函数，你会发现梯度更新的速度和𝑠𝑖𝑔𝑚𝑜𝑑函数本身的梯度是很相关的。𝑠𝑖𝑔𝑚𝑜𝑑函数在它在定义域内的梯度都不大于0.25。这样训练会非常的慢。另外，在使用𝑠𝑖𝑔𝑚𝑜𝑑函数作为正样本的概率时，同时将平方损失作为损失函数，这时所构造出来的损失函数是非凸的，不容易求解，容易得到其局部最优解。

LR的损失函数为什么要使用𝑠𝑖𝑔𝑚𝑜𝑖𝑑函数，背后的数学原理是什么？

LR假设数据服从伯努利分布,所以我们只需要知道 𝑃(𝑌|𝑋)；其次我们需要一个线性模型，所以 𝑃(𝑌|𝑋)=𝑓(𝑤𝑥)。接下来我们就只需要知道 𝑓 是什么就行了。而我们可以通过最大熵原则推出的这个𝑓，就是𝑠𝑖𝑔𝑚𝑜𝑖𝑑。
𝑠𝑖𝑔𝑚𝑜𝑖𝑑是在伯努利分布和广义线性模型的假设推导出来的。

为什么可以用梯度下降法？

因为逻辑回归的损失函数𝐿是一个连续的凸函数（conveniently convex）。这样的函数的特征是，它只会有一个全局最优的点，不存在局部最优。对于GD跟SGD最大的潜在问题就是它们可能会陷入局部最优。然而这个问题在逻辑回归里面就不存在了，因为它的损失函数的良好特性，导致它并不会有好几个局部最优。当我们的GD跟SGD收敛以后，我们得到的极值点一定就是全局最优的点，因此我们可以放心地用GD跟SGD来求解。

逻辑回归在训练的过程当中，如果有很多的特征高度相关或者说有一个特征重复了很多遍，会造成怎样的影响

如果在损失函数最终收敛的情况下，其实就算有很多特征高度相关也不会影响分类器的效果。 但是对特征本身来说的话，假设只有一个特征，在不考虑采样的情况下，你现在将它重复 N 遍。训练以后完以后，数据还是这么多，但是这个特征本身重复了 N 遍，实质上将原来的特征分成了 N 份，每一个特征都是原来特征权重值的百分之一。

为什么LR的输入特征一般是离散的而不是连续的？

在工业界，很少直接将连续值作为逻辑回归模型的特征输入，而是将连续特征离散化为一系列0、1特征交给逻辑回归模型，这样做的优势有以下几点：

离散特征的增加和减少都很容易，易于模型的快速迭代；
稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展；
离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄>30是1，否则0。如果特征没有离散化，一个异常数据“年龄300岁”会给模型造成很大的干扰；
逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合；
离散化后可以进行特征交叉，由M+N个变量变为M*N个变量，进一步引入非线性，提升表达能力；
特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问；
特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。


## 朴素贝叶斯

一、Naïve Bayes Classifier简介
贝叶斯分类器是一类分类算法的总称，贝叶斯定理是这类算法的核心，因此统称为贝叶斯分类(Bayesian Classifier)。贝叶斯决策论通过相关概率已知的情况下利用误判损失来选择最优的类别分类。一提到贝叶斯，一定少不了贝叶斯公式：
　

　

贝叶斯分类器的应用背景是：
在很多应用中，属性集和类变量之间的关系是不确定。换句话说，尽管测试记录的属性集和某些训练样例相同。但是也不能确定地预测它的类标号。这种情况产生的原因可能噪声或者出现了某些影响分类的混淆因素却没有包含在分析中。例如，考虑根据一个人的饮食和锻炼的频率来预测他是否有患心脏病的危险。尽管大多数饮食健康、经常锻炼身体的人患心脏病的机率较小，但仍有人由于遗传、过量抽烟、酗酒等其他原因而患病。确定个人的饮食是否健康、体育锻炼是否充分也还是需要论证的课题，这反过来也会给学习问题带来不确定性。贝叶斯分类器，就是这样一种对属性集和类变量的概率关系建模的方法。贝叶斯定理(Bayes theorem),它是一种把类的先验知识和从数据中收集的新证据相结合的统计原理;
本篇博文主要介绍的是朴素贝叶斯分类器（Naïve Bayes Classifier）。

二、Naïve Bayes Classifier理解
2.1 Naïve Bayes公式理解
首先看一个例子，来理解最常见的Naïve Bayes：

一个饭店，所有来吃饭的客人中，会有10%的人喝酒 —— P(B)，所有客人中，会有20%的人驾车前来—— P(A)，开车来的客人中，会有5%喝酒 —— P(B|A)。那么请问，在这个饭店喝过酒的人里，仍然会开车的比例—— P(A|B)是多少？


接下来，将上面的公式化为更一般的贝叶斯公式:假设事件 A 本身又包含多种可能性，即 A 是一个集合：A={A1,A2,…,An}A={A1,A2,…,An}，那么对于集合中任意的 Ai，贝叶斯定理可用下式表示：
　

　

接下来再通过一个例子来，理解上面的式子：

某 AI 公司招聘工程师，来了8名应聘者，这8个人里，有5个人是985院校毕业的，另外3人不是。面试官拿出一道算法题准备考察他们。根据以前的面试经验，面试官知道：985毕业生做对这道题的概率是80%，非985毕业生做对率只有30%。现在，面试管从8个人里随手指了一个人——小甲，让 TA 出来做题。结果小甲做对了，那么请问，小甲是985院校毕业的概率是多大？
现在我们来看，这道题里面的小甲的毕业院校有两种可能，也就是 A={A1,A2}
A1 —— 被选中的人是985毕业的；
A2 —— 被选中的人不是985毕业的。
B —— 被选中的人做对了面试题
P(A1) = 5/8
P(A2) = 3/8
P(B|A1) = 80% = 0.8（985毕业生做对该道面试题的先验概率）
P(B|A2) = 30% = 0.3（非985毕业生做对该道面试题的先验概率）
因此：

所以，小甲是985毕业的概率是81.6%

2.2 Naïve Bayes Classifier详细理解
2.2.1 条件独立假设
给定类标号y，朴素贝叶斯分类器在估计类条件概率时假设属性之间条件独立。条件独立假设可形式化地表述为：
　

　

其中每个属性集包含d个属性：
　

　

2.2.2 Naïve Bayes Classifier工作原理
接下来有了条件独立假设，就不必计算X的每个组合的 类条件概率，只需要对给定的Y，计算每一个Xi的条件概率。后一个方法更实用，因为他不需要很大的训练集就可以获得较好的概率估计。
给定一个未知的数据样本X, 分类法将预测X属于具有最高后验概率的类. 即, 未知的样本分配给类yj, 当且仅当

根据贝叶斯定理, 有

由于P(X) 对于所有类为常数, 只需要最大化P(X|yj)P(yj)即可。

估计P(yj)
类yj的先验概率可以用式子估计 P(yj)=nj/n（ 其中, nj是类yj中的训练样本数,而n是训练样本总数 ）
估计P(X|yj)
为便于估计P(X|yj), 假定类条件独立----给定样本的类标号, 假定属性值条件地相互独立.
于是, P(X|Y=yj)可以用下式估计

其中, P(xi |yj)可以由训练样本估值
估计P(xi |yj)
设第i个属性Ai是分类属性, 则
P(xi|yj) = nij/nj
其中nij是在属性Ai上具有值xi的yj类的训练样本数, 而nj是yj类的训练样本数
朴素贝叶斯分类器所需要的信息

计算每个类的先验概率P(yj)
P(yj)=nj/n
其中, nj是yi类的训练样本数,而n是训练样本总数
对于离散属性Ai，设的不同值为ai1, ai2, …,ail ，
对于每个类yj，计算后验概率P(aik|yj), 1<= k <= l
P(aik|yj)= nikj/nj
其中nikj 是在属性Ai上具有值aik 的yj类的训练样本数, 而nj是yj类的训练样本数
对于连续属性Ai 和每个类yj，计算yj类样本的均值(Mean)ij,标准差(Standard Deviation)ij
2.2.3 Naïve Bayes Classifier举例
如下表，判断贷款者是否具有拖欠贷款的行为：
　

　



如果一个条件概率 P(Xi=xi |Y=yj) 是0, 那么整个表达式的结果也是0
Original: P(Xi=xi |Y=yj) = nij/nj
为了 避免计算概率值为0的问题，可以使用拉普拉斯概率估计（拉普拉斯平滑）：
　

　

三、Naïve Bayes Classifier的特征以及优缺点
3.1 特征
1 面对孤立的噪声点，朴素贝叶斯分类器是健壮的。因为在从数据中估计条件概率时，这些点被平均。通过在建模和分类时忽略样例，朴素贝叶斯分类器也可以处理属性值遗漏问题。
2 面对无关属性，该分类器是健壮的。如果Xi是无关属性，那么P(XiIY)几乎变成了均匀分布。Xi的类条件概率不会对总的后验概率的计算产生影响。
3 相关属性可能会降低朴素贝叶斯分类器的性能，因为对这些属性，条件独立的假设已不成立。
3.2 优点和缺点 Strengths and Weaknesses
朴素贝叶斯分类器与其他方法相比最大的优势或许就在于，它在接受大数据量训练和查询时所具备的高速度。即使选用超大规模的训练集，针对每个项目通常也只会有相对较少的特征数，并且对项目的训练和分类也仅仅是针对特征概率的数学运算而已。

尤其当训练量逐渐递增时则更是如此。在不借助任何旧有训练数据的前提下，每一组新的训练数据都有可能会引起概率值的变化。

（你会注意到，贝叶斯分类器的算法实现代码允许我们每次只使用一个训练项，而其他方法，比如决策树和支持向量机，则须要我们一次性将整个数据集都传给它们。）对于一个如垃圾邮件过滤这样的应用程序而言，支持增量式训练的能力是非常重要的，因为过滤程序时常要对新到的邮件进行训练，然后必须即刻进行相应的调整；更何况，过滤程序也未必有权访问已经收到的所有邮件信息。

朴素贝叶斯分类器的另一大优势是，对分类器实际学习状况的解释还是相对简单的。由于每个特征的概率值都被保存了起来，因此我们可以在任何时候查看数据库，找到最适合的特征来区分垃圾邮件与非垃圾邮件，或是编程语言与蛇。保存在数据库中的这些信息都很有价值，它们有可能会被用于其他的应用程序，或者作为构筑这些应用程序的一个良好基础。

朴素贝叶斯分类器的最大缺陷就是，它无法处理基于特征组合所产生的变化结果。假设有如下这样一个场景，我们正在尝试从非垃圾邮件中鉴别出垃圾邮件来：假如我们构建的是一个Web应用程序，因而单词"online"时常会出现在你的工作邮件中。而你的好友则在一家药店工作，并且喜欢给你发一些他碰巧在工作中遇到的奇闻趣事。同时，和大多数不善于严密保护自己邮件地址的人一样，偶尔你也会收到一封包含单词"online pharmacy"的垃圾邮件。
也许你已经看出了此处的难点–我们往往会告诉分类器"online"和"pharmacy"是出现在非垃圾邮件中的，因此这些单词相对于非垃圾邮件的概率会更高一些。当我们告诉分类器有一封包含单词"online pharmacy"的邮件属于垃圾邮件时，则这些单词的概率又会进行相应的调整，这就导致了一个经常性的矛盾。由于特征的概率都是单独给出的，因此分类器对于各种组合的情况一无所知。在文档分类中，这通常不是什么大问题，因为一封包含单词"online pharmacy"的邮件中可能还会有其他特征可以说明它是垃圾邮件，但是在面对其他问题时，理解特征的组合可能是至关重要的。

四、Naïve Bayes Classifier根据姓名判断男女
主要思路：日常从一个人的名字中，基本上能大致判断这个名字的主人是男是女。比如李大志，这个名字一听就很男性。为什么呢？因为大字和志字男性名字用得比较多。虽然机器一眼看不出来，但它可以通过统计信息来判断。如果有足够多的数据，就可以统计出大字和志字用作男性名字的比例，计算概率信息。然后就可以用这些概率，运用上述的贝叶斯公式来进行计算，判定性别。
代码其实不难，各个字的统计数据已经计算好，在数据集中给出。只需要读取文件数据，存储到 python 的字典中，计算出概率，然后预测的时候进行计算即可。
举个栗子，要判断“翟天临”博士是男还是女：

P(gender=男|name=天临)
= P(name=天临|gender=男) * P(gender=男) / P(name=天临)
= P(name has 天|gender=男) * P(name has 临|gender=男) * P(gender=男) / P(name=天临

公式原理为贝叶斯公式，下面对公式中中各个项进行解答:
首先明确我们已经统计得到P(gender=男),P(gender=女)的概率。
怎么算 P(name has 天|gender=男)?
P(name has 天|gender=男)=“天”在男性名字中出现的次数 / 男性字出现的总次数
怎么算 P(gender=男)?
P(gender=男)=男性名出现的次数 / 总次数
怎么算 P(name=天临)?
这个概率对男女来说都是一样的，所以没必要算出来，即我们只需要比较P(name=天临|gender=男) * P(gender=男)和P(name=天临|gender=女) * P(gender=女)两部分谁比较大即可做出判断，并进行选择即可

代码如下（用到的男女姓名数据集在这里。）：

 ```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
__all__ = ['guess']

def compatNameCharacter(name):
    try:
        name = name.decode('utf-8')
    except:
        pass
    return name

class Guesser(object):
    #step1:init class，一上来，先初始化变量
    def __init__(self):
        self._load_model()
    #待初始化的变量
    def _load_model(self):
        self.male_total = 0
        self.female_total = 0
        self.freq = {}

        with open(os.path.join(os.path.dirname(__file__),'charfreq.csv'),'rb') as f:
            # skip first line第一行是表头，需要跳过
            next(f)
            for line in f:
                line = line.decode('utf-8')
                char, male, female = line.split(',')
         		#完成数据格式转换
                char = compatNameCharacter(char)
                self.male_total += int(male)
                self.female_total += int(female)
                self.freq[char] = (int(female), int(male))


        self.total = self.male_total + self.female_total
        # print(self.total)
        # print(self.male_total)
        # print(self.female_total)
        for char in self.freq:
            female, male = self.freq[char]
            self.freq[char] = (1. * female / self.female_total,
                               1. * male / self.male_total)
        #print(self.freq) #step1：分析每个字在男女名字的占比
    def guess(self, name):
        name = compatNameCharacter(name)
        #去掉姓，默认是单姓
        firstname = name[1:]
        for char in firstname:
            assert u'\u4e00' <= char <= u'\u9fa0', u'姓名必须为中文'
        pf = self.prob_for_gender(firstname, 0)

        print('------------')
        pm = self.prob_for_gender(firstname, 1)
        #step4:最后用男女的对比算的占比推断出性别的概率
        if pm > pf:
            return ('male', 1. * pm / (pm + pf))
        elif pm < pf:
            return ('female', 1. * pf / (pm + pf))
        else:
            return ('unknown', 0)

    def prob_for_gender(self, firstname, gender=0):
        p = 1. * self.female_total / self.total \
            if gender == 0 \
            else 1. * self.male_total / self.total
        print(p)#step2：p为男女的概率
        for char in firstname:
            p *= self.freq.get(char, (0, 0))[gender] #step3：每个字在男女总字数里出现的概率
            print(char)
            print(p)
        return p

def guess(name):
	guesser = Guesser()
    return guesser.guess(name)

if __name__ == '__main__':
    print(guess("翟天临"))
   ```

输出结果为

天
0.0014011353402999154
临
9.803745729757269e-08
('male', 0.8545506688655845)


所以，叫“天临”的有85.5%的大概率是男同志。

五、总结
朴素贝叶斯分类是一种十分简单的分类算法，叫它朴素贝叶斯分类是因为这种方法的思想真的很朴素。朴素贝叶斯的思想基础是这样的：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。尽管这些特征相互依赖或者有些特征由其他特征决定，然而朴素贝叶斯分类器认为这些属性是独立的（独立性假设）。对于某些类型的概率模型，在监督式学习的样本集中能获取得非常好的分类效果。在许多实际应用中，朴素贝叶斯模型参数估计使用最大似然估计方法；换而言之，在不用到贝叶斯概率或者任何贝叶斯模型的情况下，朴素贝叶斯模型也能奏效。
此分类器的准确率，其实是比较依赖于训练语料的，机器学习算法就和纯洁的小孩一样，取决于其训练条件,如果数据越具有代表性、质量越高，则其训练效果就越好。



# week 15: [unsupervised clustering algorithms](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%203%20Week%2015.ipynb)

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

#### K值

可以使用手肘法（The Elbow Method）：

在合理范围内，对WCSS组内平方和生成的图像进行选点

K设置的越大，样本划分越细致，每个聚类聚合程度越高，WCSS组内平方和越小

K设置的越小，样本划分越梳离，每个聚类聚合程度越低，WCSS组内平方和越大


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

# week 16: [Neural Networks](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%204%20Week%2016.ipynb)

[solution](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%204%20Week%2016%20answers.ipynb)

# 深度学习

## 神经网络（Artificial Neural Network）

1. 神经元（Artificial Neural）： 神经元是神经网络的基础·

2. 单层神经网络

在输入层和输出层之间加入了隐藏层

3. 多层神经网络（MLP）

在输入层和输出层之间加入了多个隐藏层

4. 深度神经网络（DNN）

在输入层和输出层之间加入了很多个隐藏层

5. 网络训练（Loss Function）

首先通过目前的参数进行预测，然后对其预测结果的正确性进行反馈，然后根据反馈更新参数

分类问题可以通过通过损失函数来进行操作

概率问题可以通过Softmax 函数进行操作


### [卷积神经网络（CNN）](https://zhuanlan.zhihu.com/p/47184529)

## **1. 卷积神经网络结构介绍**

如果用全连接神经网络处理大尺寸图像具有三个明显的缺点：

（1）首先将图像展开为向量会丢失空间信息；

（2）其次参数过多效率低下，训练困难；

（3）同时大量的参数也很快会导致网络过拟合。

而使用卷积神经网络可以很好地解决上面的三个问题。

与常规神经网络不同，卷积神经网络的各层中的神经元是3维排列的：宽度、高度和深度。其中的宽度和高度是很好理解的，因为本身卷积就是一个二维模板，但是在卷积神经网络中的深度指的是**激活数据体**的第三个维度，而不是整个网络的深度，整个网络的深度指的是网络的层数。举个例子来理解什么是宽度，高度和深度，假如使用CIFAR-10中的图像是作为卷积神经网络的输入，该**输入数据体**的维度是32x32x3（宽度，高度和深度）。**我们将看到，层中的神经元将只与前一层中的一小块区域连接，而不是采取全连接方式。**对于用来分类CIFAR-10中的图像的卷积网络，其最后的输出层的维度是1x1x10，因为在卷积神经网络结构的最后部分将会把全尺寸的图像压缩为包含分类评分的一个向量，**向量是在深度方向排列的**。下面是例子：

![img](https://pic3.zhimg.com/80/v2-ff46cd1067d97a86f5c2617e58c95442_1440w.jpg)图 1. 全连接神经网络与卷积神经网络的对比





图1中左侧是一个3层的神经网络；右侧是一个卷积神经网络，将它的神经元在成3个维度（宽、高和深度）进行排列。卷积神经网络的每一层都将3D的输入数据变化为神经元3D的激活数据并输出。在图1的右侧，红色的输入层代表输入图像，所以它的宽度和高度就是图像的宽度和高度，它的深度是3（代表了红、绿、蓝3种颜色通道），与红色相邻的蓝色部分是经过卷积和池化之后的激活值（也可以看做是神经元） ，后面是接着的卷积池化层。

## **2. 构建卷积神经网络的各种层**

卷积神经网络主要由这几类层构成：输入层、卷积层，ReLU层、池化（Pooling）层和全连接层（全连接层和常规神经网络中的一样）。通过将这些层叠加起来，就可以构建一个完整的卷积神经网络。在实际应用中往往将卷积层与ReLU层共同称之为卷积层，**所以卷积层经过卷积操作也是要经过激活函数的**。具体说来，卷积层和全连接层（CONV/FC）对输入执行变换操作的时候，不仅会用到激活函数，还会用到很多参数，即神经元的权值w和偏差b；而ReLU层和池化层则是进行一个固定不变的函数操作。卷积层和全连接层中的参数会随着梯度下降被训练，这样卷积神经网络计算出的分类评分就能和训练集中的每个图像的标签吻合了。

## **2.1 卷积层**

卷积层是构建卷积神经网络的**核心层**，它产生了网络中大部分的**计算量**。注意是计算量而不是参数量。

## 2.1.1 卷积层作用

\1. **滤波器的作用或者说是卷积的作用**。卷积层的参数是有一些可学习的滤波器集合构成的。每个滤波器在空间上（宽度和高度）都比较小，**但是深度和输入数据一致**（这一点很重要，后面会具体介绍）。直观地来说，网络会让滤波器学习到当它看到某些类型的视觉特征时就激活，具体的视觉特征可能是某些方位上的边界，或者在第一层上某些颜色的斑点，甚至可以是网络更高层上的蜂巢状或者车轮状图案。

\2. **可以被看做是神经元的一个输出**。神经元只观察输入数据中的一小部分，并且和空间上左右两边的所有神经元共享参数（因为这些数字都是使用同一个滤波器得到的结果）。

\3. **降低参数的数量**。这个由于卷积具有“权值共享”这样的特性，可以降低参数数量，达到降低计算开销，防止由于参数过多而造成过拟合。

## **2.1.2 感受野（重点理解）**

在处理图像这样的高维度输入时，让每个神经元都与前一层中的所有神经元进行全连接是不现实的。相反，我们让每个神经元只与输入数据的一个局部区域连接。**该连接的空间大小叫做神经元的感受野**（receptive field），它的尺寸是一个超参数（其实就是滤波器的空间尺寸）。**在深度方向上，这个连接的大小总是和输入量的深度相等**。需要再次强调的是，我们对待空间维度（宽和高）与深度维度是不同的：连接在空间（宽高）上是局部的，但是在深度上总是和输入数据的深度一致，这一点会在下面举例具体说明。

![img](https://pic4.zhimg.com/80/v2-94792663768ebde313002cdbedb5297f_1440w.jpg)图 2. 举例说明感受野的连接及尺寸说明





在图 2 中展现的卷积神经网络的一部分，其中的红色为输入数据，假设输入数据体尺寸为[32x32x3]（比如CIFAR-10的RGB图像），如果感受野（或滤波器尺寸）是5x5，那么卷积层中的每个神经元会有输入数据体中[5x5x3]区域的权重，共5x5x3=75个权重（还要加一个偏差参数）。注意这个连接在深度维度上的大小必须为3，和输入数据体的深度一致。其中还有一点需要注意，对应一个感受野有75个权重，这75个权重是通过学习进行更新的，所以很大程度上这些权值之间是不相等（也就对于同一个卷积核，它对于与它连接的输入的每一层的权重都是独特的，不是同样的权重重复输入层层数那么多次就可以的）。在这里相当于前面的每一个层对应一个传统意义上的卷积模板，每一层与自己卷积模板做完卷积之后，再将各个层的结果加起来，再加上偏置，注意是**一个偏置**，无论输入输入数据是多少层，一个卷积核就对应一个偏置。

## **2.1.3 神经元的空间排列**

感受野讲解了卷积层中每个神经元与**输入数据体**之间的连接方式，但是尚未讨论输出数据体中神经元的数量，以及它们的排列方式。3个超参数控制着**输出数据体**的尺寸：深度（depth），步长（stride）和零填充（zero-padding）。

(1) 输出数据体的深度：它是一个超参数，和使用的滤波器的数量一致，而每个滤波器在输入数据中寻找一些不同的东西，即图像的某些特征。如图2 所示，将沿着深度方向排列、感受野相同的神经元集合称为**深度列**（depth column），也有人使用纤维（fibre）来称呼它们。

(2) 在滑动滤波器的时候，必须指定步长。当步长为1，滤波器每次移动1个像素；当步长为2，滤波器滑动时每次移动2个像素，当然步长也可以是不常用的3，或者更大的数字，但这些在实际中很少使用）。这个操作会让输出数据体在空间上变小。

(3) 有时候将输入数据体用0在边缘处进行填充是很方便的。这个零填充（zero-padding）的尺寸是一个超参数。零填充有一个良好性质，即可以控制输出数据体的空间尺寸（最常用的是用来保持输入数据体在空间上的尺寸，使得**输入和输出的宽高都相等**）。

输出数据体在空间上的尺寸 ![[公式]](https://www.zhihu.com/equation?tex=W_2%C3%97H_2%C3%97D_2) 可以通过输入数据体尺寸 ![[公式]](https://www.zhihu.com/equation?tex=W_1%C3%97H_1%C3%97D_1)，卷积层中神经元的感受野尺寸（F），步长（S），滤波器数量（K）和零填充的数量（P)计算输出出来。

![img](https://pic4.zhimg.com/80/v2-dae5834d45e0e3fb3243d2adbbb738a3_1440w.jpg)

一般说来，当步长S=1时，零填充的值是P=(F-1)/2，这样就能保证输入和输出数据体有相同的空间尺寸。

**步长的限制**：注意这些空间排列的超参数之间是相互限制的。举例说来，当输入尺寸W=10，不使用零填充 P=0，滤波器尺寸 F=3，此时步长 S=2 是行不通，因为 (W-F+2P)/S+1=(10-3+0)/2+1=4.5，结果不是整数，这就是说神经元不能整齐对称地滑过输入数据体。因此，这些超参数的设定就被认为是无效的，一个卷积神经网络库可能会报出一个错误，通过修改零填充值、修改输入数据体尺寸，或者其他什么措施来让设置合理。在后面的卷积神经网络结构小节中，读者可以看到合理地设置网络的尺寸让所有的维度都能正常工作，是相当让人头痛的事；而使用零填充和遵守其他一些设计策略将会有效解决这个问题。

## **2.1.4 权值共享**

在卷积层中权值共享是用来控制参数的数量。假如在一个卷积核中，每一个感受野采用的都是不同的权重值（卷积核的值不同），那么这样的网络中参数数量将是十分巨大的。

权值共享是基于这样的一个合理的假设：如果一个特征在计算某个空间位置 (x1,y1)(x1,y1) 的时候有用，那么它在计算另一个不同位置 (x2,y2)(x2,y2) 的时候也有用。基于这个假设，可以显著地减少参数数量。换言之，就是将深度维度上一个单独的2维切片看做深度切片（depth slice），比如一个数据体尺寸为[55x55x96]的就有96个深度切片，每个尺寸为[55x55]，其中在每个深度切片上的结果都使用同样的权重和偏差获得的。在这样的参数共享下，假如一个例子中的第一个卷积层有96个卷积核，那么就有96个不同的权重集了，一个权重集对应一个深度切片，如果卷积核的大小是 11x11的，图像是RGB 3 通道的，那么就共有96x11x11x3=34,848个不同的权重，总共有34,944个参数（因为要+96个偏差），并且在每个深度切片中的55x55 的结果使用的都是同样的参数。

在反向传播的时候，都要计算每个神经元对它的权重的梯度，但是需要把同一个深度切片上的所有神经元对权重的梯度累加，这样就得到了对共享权重的梯度。这样，每个切片只更新一个权重集。这样做的原因可以通过下面这张图进行解释

![img](https://pic1.zhimg.com/80/v2-9cb091229562146799d69b05cd2c02b8_1440w.jpg)图 3. 将卷积层用全连接层的形式表示



如上图所示，左侧的神经元是将每一个感受野展开为一列之后串联起来（就是展开排成一列，同一层神经元之间不连接）。右侧的 Deep1i 是深度为1的神经元的第 i 个， Deep2i 是深度为2的神经元的第 i 个，同一个深度的神经元的权值都是相同的，黄色的都是相同的（上面4个与下面4个的参数相同），蓝色都是相同的。所以现在回过头来看上面说的卷积神经网络的反向传播公式对梯度进行累加求和也是基于这点考虑（同一深度的不同神经元共用一组参数，所以累加）；而每个切片只更新一个权重集的原因也是这样的，因为从图3 中可以看到，**不同深度的神经元不会公用相同的权重**，所以只能更新一个权重集。

**注意**，如果在一个深度切片中的所有权重都使用同一个权重向量，那么卷积层的前向传播在每个深度切片中可以看做是在计算神经元权重和输入数据体的卷积（这就是“卷积层”名字由来）。这也是为什么总是将这些权重集合称为滤波器（filter）（或卷积核（kernel）），因为它们和输入进行了卷积。

**注意**，有时候参数共享假设可能没有意义，特别是当卷积神经网络的输入图像是一些明确的中心结构时候。这时候我们就应该期望在图片的不同位置学习到完全不同的特征（而一个卷积核滑动地与图像做卷积都是在学习相同的特征）。一个具体的例子就是输入图像是人脸，人脸一般都处于图片中心，而我们期望在不同的位置学习到不同的特征，比如眼睛特征或者头发特征可能（也应该）会在图片的不同位置被学习。在这个例子中，通常就放松参数共享的限制，将层称为局部连接层（Locally-Connected Layer）。

**在这里，我推荐给你们这个深度学习框架原理实战训练营，3天就可以带你从零构建专家级神经网络框架，点击下方插件就可以免费报名哦 ↓ ↓ ↓**

3天创造属于自己的深度学习框架



## **2.1.5 卷积层的超参数及选择**

由于参数共享，每个滤波器包含 ![[公式]](https://www.zhihu.com/equation?tex=F%E2%8B%85F%E2%8B%85D_1+) 个权重（字符的具体含义在2.1.3中有介绍），卷积层一共有 ![[公式]](https://www.zhihu.com/equation?tex=F%E2%8B%85F%E2%8B%85D_1%E2%8B%85K) 个权重和 K 个偏置。在输出数据体中，第d个深度切片（空间尺寸是 ![[公式]](https://www.zhihu.com/equation?tex=W_2%C3%97H_2) ），用第d个滤波器和输入数据进行有效卷积运算的结果（使用步长S)，最后在加上第d个偏差。

对这些超参数，常见的设置是 F=3，S=1，P=1,F=3，S=1，P=1。同时设置这些超参数也有一些约定俗成的惯例和经验，可以在下面的“卷积神经网络结构”中查看。

## **2.1.6 卷积层演示**

因为3D数据难以可视化，所以所有的数据（输入数据体是蓝色，权重数据体是红色，输出数据体是绿色）都采取将深度切片按照列的方式排列展现。输入数据体的尺寸是W1=5,H1=5,D1=3W1=5,H1=5,D1=3，卷积层参数K=2,F=3,S=2,P=1K=2,F=3,S=2,P=1。就是说，有2个滤波器，滤波器的尺寸是3⋅33⋅3，它们的步长是2。因此，输出数据体的空间尺寸是(5−3+2)/2+1=3(5−3+2)/2+1=3。注意输入数据体使用了零填充P=1P=1，所以输入数据体外边缘一圈都是0。下面的例子在绿色的输出激活数据上循环演示，展示了其中每个元素都是先通过蓝色的输入数据和红色的滤波器逐元素相乘，然后**求其总和**，最后加上偏差得来。

![img](https://pic4.zhimg.com/80/v2-e7dd00d7fda722d5f8f70a9928e95a17_1440w.jpg)图 4. 卷积层演示过程



## **2.1.7 用矩阵乘法实现卷积**

卷积运算本质上就是在滤波器和输入数据的局部区域间做点积。卷积层的常用实现方式就是利用这一点，将卷积层的前向传播变成一个巨大的矩阵乘法。

(1) 输入图像的局部区域被 im2coim2co l操作拉伸为列。比如输入是[227x227x3]，要与尺寸为11x11x3的滤波器以步长为4进行卷积，就依次取输入中的[11x11x3]数据块，然后将其拉伸为长度为11x11x3=363的列向量。重复进行这一过程，因为步长为4，所以经过卷积后的宽和高均为(227-11)/4+1=55，共有55x55=3,025个个神经元。因为每一个神经元实际上都是对应有 363 的列向量构成的感受野，即一共要从输入上取出 3025 个 363 维的列向量。所以经过im2col操作得到的输出矩阵 XcolXcol 的尺寸是[363x3025]，其中每列是拉伸的感受野。注意因为感受野之间有重叠，所以输入数据体中的数字在不同的列中可能有重复。

(2) 卷积层的权重也同样被拉伸成行。举例，如果有96个尺寸为[11x11x3]的滤波器，就生成一个矩阵WrowWrow，尺寸为[96x363]。

(3) 现在卷积的结果和进行一个大矩阵乘法 np.dot(Wrow,Xcol)np.dot(Wrow,Xcol) 是等价的了，能得到每个滤波器和每个感受野间的点积。在我们的例子中，这个操作的输出是[96x3025]，给出了每个滤波器在每个位置的点积输出。注意其中的 np.dotnp.dot计算的是矩阵乘法而不是点积。

(4) 结果最后必须被重新变为合理的输出尺寸[55x55x96]。

这个方法的缺点就是占用内存太多，因为在输入数据体中的某些值在XcolXcol中被复制了多次；优点在于矩阵乘法有非常多的高效底层实现方式（比如常用的BLAS API）。还有，同样的im2col思路可以用在池化操作中。反向传播：卷积操作的反向传播（同时对于数据和权重）还是一个卷积（但是和空间上翻转的滤波器）。使用一个1维的例子比较容易演示。**这两部分中**，不是很懂如何用矩阵的形式进行汇聚操作和反向传播。

## **2.1.8 其他形式的卷积操作**

**1x1卷积**：一些论文中使用了1x1的卷积，这个方法最早是在论文Network in Network中出现。人们刚开始看见这个1x1卷积的时候比较困惑，尤其是那些具有信号处理专业背景的人。因为信号是2维的，所以1x1卷积就没有意义。但是，在卷积神经网络中不是这样，因为这里是对3个维度进行操作，滤波器和输入数据体的深度是一样的。比如，如果输入是[32x32x3]，那么1x1卷积就是在高效地进行3维点积（因为输入深度是3个通道）；另外的一种想法是将这种卷积的结果看作是全连接层的一种实现方式，详见本文2.4.2 部分。

**扩张卷积**：最近一个研究（Fisher Yu和Vladlen Koltun的论文）给卷积层引入了一个新的叫扩张（dilation）的超参数。到目前为止，我们只讨论了卷积层滤波器是连续的情况。但是，让滤波器中元素之间有间隙也是可以的，这就叫做扩张。如图5 为进行1扩张。

![img](https://pic3.zhimg.com/80/v2-f83dcfed915be7a79eb945d045c8c67e_1440w.jpg)图 5. 扩张卷积的例子及扩张前后的叠加效果

在某些设置中，扩张卷积与正常卷积结合起来非常有用，因为在很少的层数内更快地汇集输入图片的大尺度特征。比如，如果上下重叠2个3x3的卷积层，那么第二个卷积层的神经元的感受野是输入数据体中5x5的区域（可以成这些神经元的有效感受野是5x5，如图5 所示）。如果我们对卷积进行扩张，那么这个有效感受野就会迅速增长。

## **2.2 池化层**

通常在连续的卷积层之间会周期性地插入一个池化层。它的作用是逐渐降低数据体的空间尺寸，这样的话就能减少网络中参数的数量，使得计算资源耗费变少，也能有效控制过拟合。汇聚层使用 MAX 操作，对输入数据体的每一个深度切片独立进行操作，改变它的空间尺寸。最常见的形式是汇聚层使用尺寸2x2的滤波器，以步长为2来对每个深度切片进行降采样，将其中75%的激活信息都丢掉。每个MAX操作是从4个数字中取最大值（也就是在深度切片中某个2x2的区域），深度保持不变。

汇聚层的一些公式：输入数据体尺寸 ![[公式]](https://www.zhihu.com/equation?tex=W_1%E2%8B%85H_1%E2%8B%85D_1) ，有两个超参数：空间大小FF和步长SS；输出数据体的尺寸 ![[公式]](https://www.zhihu.com/equation?tex=W_2%E2%8B%85H_2%E2%8B%85D_2) ，其中

![img](https://pic1.zhimg.com/80/v2-676560fcadd292ae64f894be09c4d45c_1440w.jpg)

这里面与之前的卷积的尺寸计算的区别主要在于两点，首先在池化的过程中基本不会进行另补充；其次池化前后深度不变。



在实践中，最大池化层通常只有两种形式：一种是F=3,S=2F=3,S=2，也叫重叠汇聚（overlapping pooling），另一个更常用的是F=2,S=2F=2,S=2。对更大感受野进行池化需要的池化尺寸也更大，而且往往对网络有破坏性。
**普通池化（General Pooling）**：除了最大池化，池化单元还可以使用其他的函数，比如平均池化（average pooling）或L-2范式池化（L2-norm pooling）。平均池化历史上比较常用，但是现在已经很少使用了。因为实践证明，最大池化的效果比平均池化要好。

**反向传播**：回顾一下反向传播的内容，其中max(x,y)函数的反向传播可以简单理解为将梯度只沿最大的数回传。因此，在向前传播经过汇聚层的时候，通常会把池中最大元素的索引记录下来（有时这个也叫作道岔（switches）），这样在反向传播的时候梯度的路由就很高效。(具体如何实现我也不是很懂)。

**不使用汇聚层**：很多人不喜欢汇聚操作，认为可以不使用它。比如在Striving for Simplicity: The All Convolutional Net一文中，提出使用一种只有重复的卷积层组成的结构，抛弃汇聚层。通过在卷积层中使用更大的步长来降低数据体的尺寸。有发现认为，在训练一个良好的生成模型时，弃用汇聚层也是很重要的。比如**变化自编码器**（VAEs：variational autoencoders）和**生成性对抗网络**（GANs：generative adversarial networks）。现在看起来，未**来的卷积网络结构中，可能会很少使用甚至不使用汇聚层**。

## **2.3 归一化层**

在卷积神经网络的结构中，提出了很多不同类型的归一化层，有时候是为了实现在生物大脑中观测到的抑制机制。但是这些层渐渐都**不再流行**，因为实践证明**它们的效果即使存在，也是极其有限的**。

## **2.4 全连接层**

这个常规神经网络中一样，它们的激活可以先用矩阵乘法，再加上偏差。

## **2.4.1 将卷积层转化成全连接层**

对于任一个卷积层，都存在一个能实现和它一样的前向传播函数的全连接层。该全连接层的权重是一个巨大的矩阵，除了某些特定块（感受野），其余部分都是零；而在非 0 部分中，大部分元素都是相等的（权值共享），具体可以参考图3。如果把全连接层转化成卷积层，以输出层的 Deep11 为例，与它有关的输入神经元只有上面四个，所以在权重矩阵中与它相乘的元素，除了它所对应的4个，剩下的均为0，这也就解释了为什么权重矩阵中有为零的部分；另外要把“将全连接层转化成卷积层”和“用矩阵乘法实现卷积”区别开，这两者是不同的，后者本身还是在计算卷积，只不过将其展开为矩阵相乘的形式，并不是”将全连接层转化成卷积层”，所以除非权重中本身有零，否则用矩阵乘法实现卷积的过程中不会出现值为0的权重。

## **2.4.2 将全连接层转化成卷积层**

任何全连接层都可以被转化为卷积层。比如，一个K=4096的全连接层，输入数据体的尺寸是 7×7×5127×7×512，这个全连接层可以被等效地看做一个F=7,P=0,S=1,K=4096,F=7,P=0,S=1,K=4096的卷积层。换句话说，就是将滤波器的尺寸设置为和输入数据体的尺寸设为一致的。因为只有一个单独的深度列覆盖并滑过输入数据体，所以输出将变成1×1×40961×1×4096，这个结果就和使用初始的那个全连接层一样了。这个实际上也很好理解，因为，对于其中的一个卷积滤波器，这个滤波器的的深度为512，也就是说，虽然这个卷积滤波器的输出只有1个，但是它的权重有7×7×5127×7×512，相当于卷积滤波器的输出为一个神经元，这个神经元与上一层的所有神经元相连接，而这样与前一层所有神经元相连接的神经元一共有4096个，这不就是一个全连接网络嘛~

在上述的两种变换中，将全连接层转化为卷积层在实际运用中更加有用。假设一个卷积神经网络的输入是224x224x3的图像，一系列的卷积层和汇聚层将图像数据变为尺寸为7x7x512的激活数据体（在AlexNet中就是这样，通过使用5个汇聚层来对输入数据进行空间上的降采样，每次尺寸下降一半，所以最终空间尺寸为224/2/2/2/2/2=7）。从这里可以看到，AlexNet使用了两个尺寸为4096的全连接层，最后一个有1000个神经元的全连接层用于计算分类评分。我们可以将这3个全连接转化为3个卷积层：

(1) 针对第一个连接区域是[7x7x512]的全连接层，令其滤波器尺寸为F=7，这样输出数据体就为[1x1x4096]了。

(2) 针对第二个全连接层，令其滤波器尺寸为F=1，这样输出数据体为[1x1x4096]。

(3) 对最后一个全连接层也做类似的，令其F=1，最终输出为[1x1x1000]。

这样做的目的是让卷积网络在一张更大的输入图片上滑动，得到多个输出，这样的转化可以让我们在单个向前传播的过程中完成上述的操作。

举个例子，如果我们想让224x224尺寸的浮窗，以步长为32在384x384的图片上滑动，把每个经停的位置都带入卷积网络，最后得到6x6个位置的类别得分。上述的把全连接层转换成卷积层的做法会更简便。如果224x224的输入图片经过卷积层和汇聚层之后得到了[7x7x512]的数组，那么，384x384的大图片直接经过同样的卷积层和汇聚层之后会得到[12x12x512]的数组（因为途径5个汇聚层，尺寸变为384/2/2/2/2/2 = 12）。然后再经过上面由3个全连接层转化得到的3个卷积层，最终得到[6x6x1000]的输出（因为(12 - 7)/1 + 1 = 6）。这个结果正是浮窗在原图经停的6x6个位置的得分！

面对384x384的图像，让（含全连接层）的初始卷积神经网络以32像素的步长独立对图像中的224x224块进行多次评价，其效果和使用把全连接层变换为卷积层后的卷积神经网络进行一次前向传播是一样的。自然，相较于使用被转化前的原始卷积神经网络对所有36个位置进行迭代计算，使用转化后的卷积神经网络进行一次前向传播计算要高效得多，因为36次计算都在共享计算资源。

这里有**几个问题**，首先**为什么是以32为步长**，如果我以**64为步长**呢？再或者如果我们**想用步长小于32（如16）的浮窗怎么办**？

首先回答其中的第一个问题。这个是因为其中一个有五个汇聚层，因为25=3225=32,也就是在原始图像上的宽或者高增加 3232 个像素，经过这些卷积和汇聚后，将变为一个像素。现在进行举例说明，虽然例子并没有32那么大的尺寸，但是意义都是一样的。假设原始图像的大小为 4×4，卷积核 F=3,S=1,P=1F=3,S=1,P=1，而较大的图像的尺寸为 8×8，假设对图像进行两层的卷积池化，在较大的图像上以步长为4进行滑动（22=422=4），如图5所示



![img](https://pic1.zhimg.com/80/v2-619aa40a7f02dcf61cda3fc968073754_1440w.jpg)图 5. 以步长为4在原始图像上滑动取出4×4窗口再计算卷积的结果

对原始图像（图5左图红框）进行卷积得到的结果是图5右图红色框内的结果，使用步长为4在较大的图像获得的结果为图5中右侧四种颜色加在一起的样子。所以以步长为4在8x8的图片上滑动，把每个经停的位置都带入卷积网络，最后得到2x2个位置的卷积结果，但是如果直接使用卷积核 F=3,S=1,P=1F=3,S=1,P=1进行两次卷积池化的话，得到的结果的大小显然也是4×4的。

所以从获得结果来看，这两者是相同的，但是不同点在哪呢？如图6所示，是在整个图像上进行卷积运算和以步长为4在8x8的图片上滑动所经停的第一个位置，这两种方法使用相同的卷积核进行计算的对比图。



![img](https://pic1.zhimg.com/80/v2-4ce114b0441bd5a803aedc94b91fb9b4_1440w.jpg)图6. 使用整张图像和一部分图像计算某一点处的卷积

如图6所示，左图代表使用整张图像时计算a点处的卷积，右图代表使用滑动的方法第一次经停图像上a点的卷积，两张图中的a点是同一个a点。虽然同一个卷积模板进行计算，但是在计算卷积的过程是不同的！因为在右图中a的右侧及右下侧是0，而在左图中是原始的像素值，所以计算的卷积一定是不同的。但是要怎么理解这样的差别呢？这要从补零的意义讲起，补零是因为如果不补零的话，图像经过卷积之后的尺寸会小于原始的尺寸，补零可以保证图像的尺寸不变，所以归根结底补零实际上是一种图像填充的方法。左图中a的右边及右下角的像素是原始图像的像素，相当于在计算a点的时候，不是用0进行的补充，而是原始像素值进行补充，这样的不仅可以保持卷积前后图像的大小不变，而且可以这种图像填充方法得到的结果显然要比填0更接近与原始图像，保留的信息更多。

**小节**

(1) 用一整图像进行卷积和在较大的图像上通过滑动窗提取出一个个子图象进行卷积得到的效果是相同的。

(2) 可以这样做的主要原因在于将最后的全连接层改写成了卷积层。

(3) 在一整章图像做卷积的效率要远远高于在图像上滑动的效率，因为前者只需要依次前向传播，而后者需要多次

(4) 用整张图像计算与滑动窗口的方法对比，所补充的零更少（如上所讲，不用零而是用在其旁边的像素代替），提取的信息损失的更少。

即，**用整张图像直接计算卷积不仅仅在效率上高于使用滑动窗口的方法，而且更多的保留了图像的细节**，完胜！

另外还可以得到**另一个结论**，**当在较大的图像上以步长为** **2L2L进行滑动时，其效果与在有直接在大图像上进行卷积得到的结果上以步长为1移动是一样的**。如图5中的大红色框对应小红色框，大黄色框对应小黄色框。所以当步长为64时，将相当于以步长为2在大图的卷积结果上移动。

对于第二个问题，如果我非要以12为步长呢？是可以的，只是这个时候所获得结果不再是如图5的那种滑动的计算方式了。还是举例说明，不过为了方便说明改变了一下尺寸。将步长32改为4，将步长16改为2进行分析。假如说原始的输入图像为一个 4×4 的图像，现在将使用一个比原来大的图像，是一个8×8的图像，使用卷积核为 4×4 大小，步长为4，则在图像进行卷积运算的如图6左侧的4个部分（红黄绿蓝），而图6 右侧的是步长为2时与原始相比增加的部分。将图 6中两个部分相加就可以得到步长为2它时所有进行卷积运算的部分了。

![img](https://pic4.zhimg.com/80/v2-2352106e15a7f179b44dba1db284573f_1440w.jpg)图6 .步长为4时原始图像进行卷积的部分及将步长改为2时比原来多出的部分

获得步长为2的时进行卷积的区域。首先像之前一样对原始图像做以4为步长的卷积，这时进行卷积的部分就是图6中左侧的部分；其次将原始图片沿宽度方向平移2个像素之后，依旧进行步长为4的卷积，这个时候进行卷积的部分为图6中的红色部分和绿色部分；然后沿高度方向平移2个像素之后，按步长为4进行卷积，这个时候进行卷积的部分为图6中的蓝色部分和黄色部分；最后沿高度方向和宽度方向同时移动2个像素，按步长为4进行卷积，这个时候进行卷积的部分为图6中的紫色部分。将这些部分加在一起就是进行卷积运算你得所有区域了。

这个结果明显是无法通过像图5中的那样滑动得到了，这样的方法所需要进行卷积的区域要远远大于以4为步长时所需要就进行卷积运算的区域；后续的卷积都是在这一卷积的结果上进行的，所以后面的都会发生改变。

综上，步长为32的正整数倍只是保证获得结果可以像图5那样滑动的获得的下限值。

## **3. 卷积神经网络的结构**

卷积神经网络通常是由三种层构成：卷积层，汇聚层（除非特别说明，一般就是最大值汇聚）和全连接层（简称FC）。ReLU激活函数也应该算是是一层，它逐元素地进行激活函数操作，常常将它与卷积层看作是同一层。

## **3.1 层的排列规律**

卷积神经网络最常见的形式就是将一些卷积层和ReLU层放在一起，其后紧跟汇聚层，然后重复如此直到图像在空间上被缩小到一个足够小的尺寸，在某个地方过渡成成全连接层也较为常见。最后的全连接层得到输出，比如分类评分等。换句话说，最常见的卷积神经网络结构如下：

![img](https://pic1.zhimg.com/80/v2-2dbb2fab147f00d3a2b0409a7a07175c_1440w.jpg)

其中*指的是重复次数，POOL?指的是一个可选的汇聚层。其中N >=0,通常N<=3,M>=0,K>=0,通常K<3。例如，下面是一些常见的网络结构规律：

- **INPUT -> FC** ，实现一个线性分类器，此处N = M = K = 0。
- **INPUT -> CONV -> RELU -> FC**，单层的卷积神经网络
- **INPUT -> [CONV -> RELU -> POOL]\*2 -> FC -> RELU -> FC**，此处在每个汇聚层之间有一个卷积层，这种网络就是简单的多层的卷积神经网络。
- **INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]\*3 -> [FC -> RELU]\*2 -> FC** ，此处每个汇聚层前有两个卷积层，这个思路适用于更大更深的网络（比如说这个思路就和VGG比较像），因为在执行具有破坏性的汇聚操作前，多重的卷积层可以从输入数据中学习到更多的复杂特征。

**最新进展**：传统的将层按照线性进行排列的方法已经受到了挑战，挑战来自谷歌的Inception结构和微软亚洲研究院的残差网络（Residual Net）结构。这两个网络的特征更加复杂，连接结构也不同。

## **3.2 卷积层的大小选择**

**几个小滤波器卷积层的组合比一个大滤波器卷积层好**。假设你一层一层地重叠了3个3x3的卷积层（层与层之间有非线性激活函数）。在这个排列下，第一个卷积层中的每个神经元都对输入数据体有一个3x3的视野。第二个卷积层上的神经元对第一个卷积层有一个3x3的视野，也就是对输入数据体有5x5的视野。同样，在第三个卷积层上的神经元对第二个卷积层有3x3的视野，也就是对输入数据体有7x7的视野。假设不采用这3个3x3的卷积层，二是使用一个单独的有7x7的感受野的卷积层，那么所有神经元的感受野也是7x7，但是就有一些缺点。首先，多个卷积层与非线性的激活层交替的结构，比单一卷积层的结构更能提取出深层的更好的特征。其次，假设所有的数据有C个通道，那么单独的7x7卷积层将会包含 ![[公式]](https://www.zhihu.com/equation?tex=C%C3%97%287%C3%977%C3%97C%29%3D49C%5E2) 个参数，而3个3x3的卷积层的组合仅有 ![[公式]](https://www.zhihu.com/equation?tex=3%C3%97%28C%C3%97%283%C3%973%C3%97C%29%29%3D27C%5E2) 个参数。直观说来，最好选择带有小滤波器的卷积层组合，而不是用一个带有大的滤波器的卷积层。前者可以表达出输入数据中更多个强力特征，使用的参数也更少。唯一的不足是，在进行反向传播时，中间的卷积层可能会导致占用更多的内存。

## **3.3 层的尺寸设置规律**

- **输入层** ，应该能被2整除很多次。常用数字包括32（比如CIFAR-10），64，96（比如STL-10）或224（比如ImageNet卷积神经网络），384和512。
- **卷积层** ，应该使用小尺寸滤波器（比如3x3或最多5x5），使用步长S=1。还有一点非常重要，就是对输入数据进行零填充，这样卷积层就不会改变输入数据在空间维度上的尺寸。比如，当F=3，那就使用P=1来保持输入尺寸。当F=5,P=2，一般对于任意F，当P=(F-1)/2的时候能保持输入尺寸。如果必须使用更大的滤波器尺寸（比如7x7之类），通常只用在第一个面对原始图像的卷积层上。
- **汇聚层** ，负责对输入数据的空间维度进行降采样。最常用的设置是用用2x2感受野（即F=2）的最大值汇聚，步长为2（S=2）。注意这一操作将会把输入数据中75%的激活数据丢弃（因为对宽度和高度都进行了2的降采样）。另一个不那么常用的设置是使用3x3的感受野，步长为2。最大值汇聚的感受野尺寸很少有超过3的，因为汇聚操作过于激烈，易造成数据信息丢失，这通常会导致算法性能变差。

上文中展示的两种设置(卷积层F=3，P=1，汇聚层F=2,P=2)是很好的，因为所有的卷积层都能保持其输入数据的空间尺寸，汇聚层只负责对数据体从空间维度进行降采样。如果使用的步长大于1并且不对卷积层的输入数据使用零填充，那么就必须非常仔细地监督输入数据体通过整个卷积神经网络结构的过程，确认所有的步长和滤波器都尺寸互相吻合，卷积神经网络的结构美妙对称地联系在一起。

**为何使用零填充**？使用零填充除了前面提到的可以让卷积层的输出数据保持和输入数据在空间维度的不变，还可以提高算法性能。如果卷积层值进行卷积而不进行零填充，那么数据体的尺寸就会略微减小，那么图像边缘的信息就会过快地损失掉。

**因为内存限制所做的妥协**：在某些案例（尤其是早期的卷积神经网络结构）中，基于前面的各种规则，内存的使用量迅速飙升。例如，使用64个尺寸为3x3的滤波器对224x224x3的图像进行卷积，零填充为1，得到的激活数据体尺寸是[224x224x64]。这个数量就是一千万的激活数据，或者就是72MB的内存（每张图就是这么多，激活函数和梯度都是）。因为GPU通常因为内存导致性能瓶颈，所以做出一些妥协是必须的。在实践中，人们倾向于在网络的第一个卷积层做出妥协。例如，可以妥协可能是在第一个卷积层使用步长为2，尺寸为7x7的滤波器（比如在ZFnet中）。在AlexNet中，滤波器的尺寸的11x11，步长为4。

## **4. 案例学习**

下面是卷积神经网络领域中比较有名的几种结构：

- **LeNet** ，第一个成功的卷积神经网络应用，是Yann LeCun在上世纪90年代实现的。当然，最著名还是被应用在识别数字和邮政编码等的LeNet结构。
- **AlexNet** ，AlexNet卷积神经网络在计算机视觉领域中受到欢迎，它由Alex Krizhevsky，Ilya Sutskever和Geoff Hinton实现。AlexNet在2012年的ImageNet ILSVRC 竞赛中夺冠，性能远远超出第二名（16%的top5错误率，第二名是26%的top5错误率）。这个网络的结构和LeNet非常类似，但是更深更大，并且使用了**层叠的卷积层**来获取特征（**之前通常是只用一个卷积层并且在其后马上跟着一个汇聚层**）。
- **ZF Net** ，Matthew Zeiler和Rob Fergus发明的网络在ILSVRC 2013比赛中夺冠，它被称为 ZFNet（Zeiler & Fergus Net的简称）。它通过修改结构中的超参数来实现对AlexNet的改良，具体说来就是**增加了中间卷积层的尺寸**，**让第一层的步长和滤波器尺寸更小**。
- **GoogLeNet** ，ILSVRC 2014的胜利者是谷歌的Szeged等实现的卷积神经网络。它主要的贡献就是实现了一个奠基模块，它能够**显著地减少网络中参数的数量**（AlexNet中有60M，该网络中只有4M）。还有，这个论文中**没有使用卷积神经网络顶部使用全连接层**，而是使用了一个平均汇聚，把大量不是很重要的参数都去除掉了。GooLeNet还有几种改进的版本，**最新的一个是Inception-v4**。
- **VGGNet** ，ILSVRC 2014的第二名是Karen Simonyan和 Andrew Zisserman实现的卷积神经网络，现在称其为VGGNet。它主要的贡献是展示出网络的**深度是算法优良性能的关键部分**。他们最好的网络包含了16个卷积/全连接层。网络的结构非常一致，**从头到尾全部使用的是3x3的卷积和2x2的汇聚**。他们的预训练模型是可以在网络上获得并在Caffe中使用的。VGGNet**不好的一点是它耗费更多计算资源，并且使用了更多的参数，导致更多的内存占用（140M）**。其中绝大多数的参数都是来自于第一个全连接层。后来发现这些**全连接层即使被去除，对于性能也没有什么影响**，这样就显著降低了参数数量。
- **ResNet** ，残差网络（Residual Network）是ILSVRC2015的胜利者，由何恺明等实现。它使用了特殊的**跳跃链接**，大量使用了**批量归一化**（batch normalization）。这个结构同样在最后没有使用全连接层。读者可以查看何恺明的的演讲（视频，PPT），以及一些使用Torch重现网络的实验。ResNet当前最好的卷积神经网络模型（2016年五月）。何开明等最近的工作是对原始结构做一些优化，可以看论文Identity Mappings in Deep Residual Networks，2016年3月发表。

## 4.1 VGGNet的细节

我们进一步对VGGNet的细节进行分析学习。整个VGGNet中的卷积层都是以步长为1进行3x3的卷积，使用了1的零填充，汇聚层都是以步长为2进行了2x2的最大值汇聚。可以写出处理过程中每一步数据体尺寸的变化，然后对数据尺寸和整体权重的数量进行查看：

```text
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000

TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters
```

注意，**大部分的内存和计算时间都被前面的卷积层占用，大部分的参数都用在后面的全连接层**，这在卷积神经网络中是比较常见的。在这个例子中，**全部参数有140M，但第一个全连接层就包含了100M的参数**。

## **5. 计算上的考量**

在构建卷积神经网络结构时，最大的瓶颈是内存瓶颈，所以如何降低内存消耗量是一个值得思考的问题。三种内存占用来源：

- **1** 来自中间数据体尺寸：卷积神经网络中的每一层中都有激活数据体的原始数值，以及损失函数对它们的梯度（和激活数据体尺寸一致）。通常，大部分激活数据都是在网络中靠前的层中（比如第一个卷积层）。在训练时，这些数据需要放在内存中，因为反向传播的时候还会用到。但是在测试时可以聪明点：让网络在测试运行时候每层都只存储当前的激活数据，然后丢弃前面层的激活数据，这样就能减少巨大的激活数据量。这实际上是底层问题，在编写框架的过程中，设计者会进行这方面的考虑。
- **2** 来自参数尺寸：即整个网络的参数的数量，在反向传播时它们的梯度值，以及使用momentum、Adagrad或RMSProp等方法进行最优化时的每一步计算缓存。因此，存储参数向量的内存通常需要在参数向量的容量基础上乘以3或者更多。
- **3** 卷积神经网络实现还有各种零散的内存占用，比如成批的训练数据，扩充的数据等等。

一旦对于所有这些数值的数量有了一个大略估计（包含激活数据，梯度和各种杂项），数量应该转化为以GB为计量单位。把这个值乘以4，得到原始的字节数（因为每个浮点数占用4个字节，如果是双精度浮点数那就是占用8个字节），然后多次除以1024分别得到占用内存的KB，MB，最后是GB计量。如果你的网络工作得不好，一个常用的方法是降低批尺寸（batch size），因为绝大多数的内存都是被激活数据消耗掉了。



# week 17: [Decision Trees](https://github.com/gwt9970161/Introduction-to-AI/blob/main/Worksheet%205%20Week%2017.ipynb)

# 决策树模型

[TOC]

决策树是一个非常常见并且优秀的机器学习算法，它易于理解、可解释性强，其可作为分类算法，也可用于回归模型。本文将分三篇介绍决策树，第一篇介绍基本树（包括 ID3、C4.5、CART），第二篇介绍 Random Forest、Adaboost、GBDT，第三篇介绍 Xgboost 和 LightGBM。

对于基本树我将大致从以下四个方面介绍每一个算法：思想、划分标准、剪枝策略，优缺点。

## 1. ID3

ID3 算法是建立在奥卡姆剃刀（用较少的东西，同样可以做好事情）的基础上：越是小型的决策树越优于大的决策树。

### 1.1 思想

从信息论的知识中我们知道：信息熵越大，从而样本纯度越低，。ID3 算法的核心思想就是以信息增益来度量特征选择，选择信息增益最大的特征进行分裂。算法采用自顶向下的贪婪搜索遍历可能的决策树空间（C4.5 也是贪婪搜索）。 其大致步骤为：

1. 初始化特征集合和数据集合；
2. 计算数据集合信息熵和所有特征的条件熵，选择信息增益最大的特征作为当前决策节点；
3. 更新数据集合和特征集合（删除上一步使用的特征，并按照特征值来划分不同分支的数据集合）；
4. 重复 2，3 两步，若子集值包含单一特征，则为分支叶子节点。

### 1.2 划分标准

ID3 使用的分类标准是信息增益，它表示得知特征 A 的信息而使得样本集合不确定性减少的程度。

数据集的信息熵：

![[公式]](https://www.zhihu.com/equation?tex=H%28D%29%3D-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7Dlog_2%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=C_k) 表示集合 D 中属于第 k 类样本的样本子集。

针对某个特征 A，对于数据集 D 的条件熵 ![[公式]](https://www.zhihu.com/equation?tex=H%28D%7CA%29) 为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+H%28D%7CA%29+%26+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7DH%28D_i%29+%5C%5C+%26+%3D-+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7D%28%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7Dlog_2%5Cfrac%7B%7CD_%7Bik%7D%7C%7D%7B%7CD_i%7C%7D%29++%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=D_i) 表示 D 中特征 A 取第 i 个值的样本子集， ![[公式]](https://www.zhihu.com/equation?tex=D_%7Bik%7D) 表示 ![[公式]](https://www.zhihu.com/equation?tex=D_i) 中属于第 k 类的样本子集。

信息增益 = 信息熵 - 条件熵：

![[公式]](https://www.zhihu.com/equation?tex=Gain%28D%2CA%29%3DH%28D%29-H%28D%7CA%29++%5C%5C)

信息增益越大表示使用特征 A 来划分所获得的“纯度提升越大”。

### 1.3 缺点

- ID3 没有剪枝策略，容易过拟合；
- 信息增益准则对可取值数目较多的特征有所偏好，类似“编号”的特征其信息增益接近于 1；
- 只能用于处理离散分布的特征；
- 没有考虑缺失值。

## 2. C4.5

C4.5 算法最大的特点是克服了 ID3 对特征数目的偏重这一缺点，引入信息增益率来作为分类标准。

### 2.1 思想

C4.5 相对于 ID3 的缺点对应有以下改进方式：

- 引入悲观剪枝策略进行后剪枝；
- 引入信息增益率作为划分标准；
- 将连续特征离散化，假设 n 个样本的连续特征 A 有 m 个取值，C4.5 将其排序并取相邻两样本值的平均数共 m-1 个划分点，分别计算以该划分点作为二元分类点时的信息增益，并选择信息增益最大的点作为该连续特征的二元离散分类点；
- 对于缺失值的处理可以分为两个子问题：
- 问题一：在特征值缺失的情况下进行划分特征的选择？（即如何计算特征的信息增益率）
- 问题二：选定该划分特征，对于缺失该特征值的样本如何处理？（即到底把这个样本划分到哪个结点里）
- 针对问题一，C4.5 的做法是：对于具有缺失值特征，用没有缺失的样本子集所占比重来折算；
- 针对问题二，C4.5 的做法是：将样本同时划分到所有子节点，不过要调整样本的权重值，其实也就是以不同概率划分到不同节点中。

### 2.2 划分标准

利用信息增益率可以克服信息增益的缺点，其公式为

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+Gain_%7Bratio%7D%28D%2CA%29%26%3D%5Cfrac%7BGain%28D%2CA%29%7D%7BH_A%28D%29%7D+%5C%5C+H_A%28D%29+%26%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7Dlog_2%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7D+%5Cend%7Baligned%7D+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=H_A%28D%29+) 称为特征 A 的固有值。

这里需要注意，信息增益率对可取值较少的特征有所偏好（分母越小，整体越大），因此 C4.5 并不是直接用增益率最大的特征进行划分，而是使用一个**启发式方法**：先从候选划分特征中找到信息增益高于平均值的特征，再从中选择增益率最高的。

### 2.3 剪枝策略

为什么要剪枝：过拟合的树在泛化能力的表现非常差。

**2.3.1 预剪枝**

在节点划分前来确定是否继续增长，及早停止增长的主要方法有：

- 节点内数据样本低于某一阈值；
- 所有节点特征都已分裂；
- 节点划分前准确率比划分后准确率高。

预剪枝不仅可以降低过拟合的风险而且还可以减少训练时间，但另一方面它是基于“贪心”策略，会带来欠拟合风险。

**2.3.2 后剪枝**

在已经生成的决策树上进行剪枝，从而得到简化版的剪枝决策树。

C4.5 采用的**悲观剪枝方法**，用递归的方式从低往上针对每一个非叶子节点，评估用一个最佳叶子节点去代替这课子树是否有益。如果剪枝后与剪枝前相比其错误率是保持或者下降，则这棵子树就可以被替换掉。C4.5 通过训练数据集上的错误分类数量来估算未知样本上的错误率。

后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树。但同时其训练时间会大的多。

### 2.4 缺点

- 剪枝策略可以再优化；
- C4.5 用的是多叉树，用二叉树效率更高；
- C4.5 只能用于分类；
- C4.5 使用的熵模型拥有大量耗时的对数运算，连续值还有排序运算；
- C4.5 在构造树的过程中，对数值属性值需要按照其大小进行排序，从中选择一个分割点，所以只适合于能够驻留于内存的数据集，当训练集大得无法在内存容纳时，程序无法运行。

## 3. CART

ID3 和 C4.5 虽然在对训练样本集的学习中可以尽可能多地挖掘信息，但是其生成的决策树分支、规模都比较大，CART 算法的二分法可以简化决策树的规模，提高生成决策树的效率。

### 3.1 思想

CART 包含的基本过程有分裂，剪枝和树选择。

- **分裂：**分裂过程是一个二叉递归划分过程，其输入和预测特征既可以是连续型的也可以是离散型的，CART 没有停止准则，会一直生长下去；
- **剪枝：**采用**代价复杂度剪枝**，从最大树开始，每次选择训练数据熵对整体性能贡献最小的那个分裂节点作为下一个剪枝对象，直到只剩下根节点。CART 会产生一系列嵌套的剪枝树，需要从中选出一颗最优的决策树；
- **树选择：**用单独的测试集评估每棵剪枝树的预测性能（也可以用交叉验证）。

CART 在 C4.5 的基础上进行了很多提升。

- C4.5 为多叉树，运算速度慢，CART 为二叉树，运算速度快；
- C4.5 只能分类，CART 既可以分类也可以回归；
- CART 使用 Gini 系数作为变量的不纯度量，减少了大量的对数运算；
- CART 采用代理测试来估计缺失值，而 C4.5 以不同概率划分到不同节点中；
- CART 采用“基于代价复杂度剪枝”方法进行剪枝，而 C4.5 采用悲观剪枝方法。

### 3.2 划分标准

熵模型拥有大量耗时的对数运算，基尼指数在简化模型的同时还保留了熵模型的优点。基尼指数代表了模型的不纯度，基尼系数越小，不纯度越低，特征越好。这和信息增益（率）正好相反。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+Gini%28D%29%26%3D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%281-%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%29+%5C%5C+%26%3D1-%5Csum_%7Bk%3D1%7D%5E%7BK%7D%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%29%5E2++%5C%5C++Gini%28D%7CA%29++%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cfrac%7B%7CD_i%7C%7D%7B%7CD%7C%7DGini%28D_i%29+%5Cend%7Baligned%7D+%5C%5C)

其中 k 代表类别。

基尼指数反映了从**数据集中随机抽取两个样本，其类别标记不一致的概率**。因此基尼指数越小，则数据集纯度越高。基尼指数偏向于特征值较多的特征，类似信息增益。基尼指数可以用来度量任何不均匀分布，是介于 0~1 之间的数，0 是完全相等，1 是完全不相等，

此外，当 CART 为二分类，其表达式为：

![[公式]](https://www.zhihu.com/equation?tex=Gini%28D%7CA%29%3D%5Cfrac%7B%7CD_1%7C%7D%7B%7CD%7C%7DGini%28D_1%29%2B%5Cfrac%7B%7CD_2%7C%7D%7B%7CD%7C%7DGini%28D_2%29++%5C%5C)

我们可以看到在平方运算和二分类的情况下，其运算更加简单。当然其性能也与熵模型非常接近。

那么问题来了：基尼指数与熵模型性能接近，但到底与熵模型的差距有多大呢？

我们知道 ![[公式]](https://www.zhihu.com/equation?tex=ln%28x%29+%3D+-1%2Bx+%2Bo%28x%29) ，所以

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+H%28X%29%26%3D-%5Csum_%7Bk%3D1%7D%5E%7BK%7D+p_%7Bk%7D+%5Cln+p_%7Bk%7D%5C%5C%26%5Capprox+%5Csum_%7Bk%3D1%7D%5E%7BK%7D+p_%7Bk%7D%5Cleft%281-p_%7Bk%7D%5Cright%29+%5Cend%7Baligned%7D+%5C%5C)

我们可以看到，基尼指数可以理解为熵模型的一阶泰勒展开。这边在放上一张很经典的图：

![img](https://pic2.zhimg.com/80/v2-cc5fb97eb85632fa7b930baffdae0769_1440w.jpg)

### 3.3 缺失值处理

上文说到，模型对于缺失值的处理会分为两个子问题：

1. 如何在特征值缺失的情况下进行划分特征的选择？
2. 选定该划分特征，模型对于缺失该特征值的样本该进行怎样处理？

对于问题 1，CART 一开始严格要求分裂特征评估时只能使用在该特征上没有缺失值的那部分数据，在后续版本中，CART 算法使用了一种惩罚机制来抑制提升值，从而反映出缺失值的影响（例如，如果一个特征在节点的 20% 的记录是缺失的，那么这个特征就会减少 20% 或者其他数值）。

对于问题 2，CART 算法的机制是为树的每个节点都找到代理分裂器，无论在训练数据上得到的树是否有缺失值都会这样做。在代理分裂器中，特征的分值必须超过默认规则的性能才有资格作为代理（即代理就是代替缺失值特征作为划分特征的特征），当 CART 树中遇到缺失值时，这个实例划分到左边还是右边是决定于其排名最高的代理，如果这个代理的值也缺失了，那么就使用排名第二的代理，以此类推，如果所有代理值都缺失，那么默认规则就是把样本划分到较大的那个子节点。代理分裂器可以确保无缺失训练数据上得到的树可以用来处理包含确实值的新数据。

### 3.4 剪枝策略

采用一种“基于代价复杂度的剪枝”方法进行后剪枝，这种方法会生成一系列树，每个树都是通过将前面的树的某个或某些子树替换成一个叶节点而得到的，这一系列树中的最后一棵树仅含一个用来预测类别的叶节点。然后用一种成本复杂度的度量准则来判断哪棵子树应该被一个预测类别值的叶节点所代替。这种方法需要使用一个单独的测试数据集来评估所有的树，根据它们在测试数据集熵的分类性能选出最佳的树。

我们来看具体看一下代价复杂度剪枝算法：

首先我们将最大树称为 ![[公式]](https://www.zhihu.com/equation?tex=T_0) ，我们希望减少树的大小来防止过拟合，但又担心去掉节点后预测误差会增大，所以我们定义了一个损失函数来达到这两个变量之间的平衡。损失函数定义如下：

![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28T%29%3DC%28T%29%2B%5Calpha%7CT%7C++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=T) 为任意子树， ![[公式]](https://www.zhihu.com/equation?tex=C%28T%29) 为预测误差， ![[公式]](https://www.zhihu.com/equation?tex=%7CT%7C) 为子树 ![[公式]](https://www.zhihu.com/equation?tex=T) 的叶子节点个数， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是参数， ![[公式]](https://www.zhihu.com/equation?tex=C%28T%29) 衡量训练数据的拟合程度， ![[公式]](https://www.zhihu.com/equation?tex=%7CT%7C) 衡量树的复杂度， ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 权衡拟合程度与树的复杂度。

那么如何找到合适的 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 来使得复杂度和拟合度达到最好的平衡点呢，最好的办法就是另 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 从 0 取到正无穷，对于每一个固定的 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，我们都可以找到使得 ![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28T%29) 最小的最优子树 ![[公式]](https://www.zhihu.com/equation?tex=T%28%5Calpha%29) 。当 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha+) 很小的时候， ![[公式]](https://www.zhihu.com/equation?tex=T_0) 是最优子树；当 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 最大时，单独的根节点是这样的最优子树。随着 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 增大，我们可以得到一个这样的子树序列： ![[公式]](https://www.zhihu.com/equation?tex=T_0%2C+T_1%2C+T_2%2C+T_3%2C+...+%2CT_n) ，这里的子树 ![[公式]](https://www.zhihu.com/equation?tex=T_%7Bi%2B1%7D) 生成是根据前一个子树 ![[公式]](https://www.zhihu.com/equation?tex=T_i) 剪掉某一个内部节点生成的。

Breiman 证明：将 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 从小增大， ![[公式]](https://www.zhihu.com/equation?tex=0%3D%5Calpha_0%3C%5Calpha_0%3C...%3C%5Calpha_n%3C%5Cinfty) ，在每个区间 ![[公式]](https://www.zhihu.com/equation?tex=%5B%5Calpha_i%2C%5Calpha_%7Bi%2B1%7D%29) 中，子树 ![[公式]](https://www.zhihu.com/equation?tex=T_i) 是这个区间里最优的。

这是代价复杂度剪枝的核心思想。

我们每次剪枝都是针对某个非叶节点，其他节点不变，所以我们只需要计算该节点剪枝前和剪枝后的损失函数即可。

对于任意内部节点 t，剪枝前的状态，有 ![[公式]](https://www.zhihu.com/equation?tex=%7CT_t%7C) 个叶子节点，预测误差是 ![[公式]](https://www.zhihu.com/equation?tex=C%28T_t%29) ；剪枝后的状态：只有本身一个叶子节点，预测误差是 ![[公式]](https://www.zhihu.com/equation?tex=C%28t%29) 。

因此剪枝前以 t 节点为根节点的子树的损失函数是：

![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28T%29%3DC%28T_t%29%2B%5Calpha%7CT%7C+%5C%5C)

剪枝后的损失函数是

![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28t%29+%3D+C%28t%29%2B%5Calpha+%5C%5C)

通过 Breiman 证明我们知道一定存在一个 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 使得 ![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28T%29%3DC_%5Calpha%28t%29) ，使得这个值为：

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha+%3D+%5Cfrac%7BC%28t%29-C%28T_t%29%7D%7B%7CT_t%7C-1%7D++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 的意义在于， ![[公式]](https://www.zhihu.com/equation?tex=%5B%5Calpha_i%2C%5Calpha_%7Bi%2B1%7D%29) 中，子树 ![[公式]](https://www.zhihu.com/equation?tex=T_i) 是这个区间里最优的。当 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 大于这个值是，一定有 ![[公式]](https://www.zhihu.com/equation?tex=C_%5Calpha%28T%29%3EC_%5Calpha%28t%29) ，也就是剪掉这个节点后都比不剪掉要更优。所以每个最优子树对应的是一个区间，在这个区间内都是最优的。

然后我们对 ![[公式]](https://www.zhihu.com/equation?tex=T_i) 中的每个内部节点 t 都计算：

![[公式]](https://www.zhihu.com/equation?tex=g%28t%29+%3D+%5Cfrac%7BC%28t%29-C%28T_t%29%7D%7B%7CT_t%7C-1%7D+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=g%28t%29) 表示阈值，故我们每次都会减去最小的 ![[公式]](https://www.zhihu.com/equation?tex=T_t) 。

### 3.5 类别不平衡

CART 的一大优势在于：无论训练数据集有多失衡，它都可以将其子冻消除不需要建模人员采取其他操作。

CART 使用了一种先验机制，其作用相当于对类别进行加权。这种先验机制嵌入于 CART 算法判断分裂优劣的运算里，在 CART 默认的分类模式中，总是要计算每个节点关于根节点的类别频率的比值，这就相当于对数据自动重加权，对类别进行均衡。

对于一个二分类问题，节点 node 被分成类别 1 当且仅当：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BN_1%28node%29%7D%7BN_1%28root%29%7D+%3E+%5Cfrac%7BN_0%28node%29%7D%7BN_0%28root%29%7D++%5C%5C)

比如二分类，根节点属于 1 类和 0 类的分别有 20 和 80 个。在子节点上有 30 个样本，其中属于 1 类和 0 类的分别是 10 和 20 个。如果 10/20>20/80，该节点就属于 1 类。

通过这种计算方式就无需管理数据真实的类别分布。假设有 K 个目标类别，就可以确保根节点中每个类别的概率都是 1/K。这种默认的模式被称为“先验相等”。

先验设置和加权不同之处在于先验不影响每个节点中的各类别样本的数量或者份额。先验影响的是每个节点的类别赋值和树生长过程中分裂的选择。

### 3.6 回归树

CART（Classification and Regression Tree，分类回归树），从名字就可以看出其不仅可以用于分类，也可以应用于回归。其回归树的建立算法上与分类树部分相似，这里简单介绍下不同之处。

**3.6.1 连续值处理**

对于连续值的处理，CART 分类树采用基尼系数的大小来度量特征的各个划分点。在回归模型中，我们使用常见的和方差度量方式，对于任意划分特征 A，对应的任意划分点 s 两边划分成的数据集 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_2) ，求出使 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_2) 各自集合的均方差最小，同时 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=D_2) 的均方差之和最小所对应的特征和特征值划分点。表达式为：

![[公式]](https://www.zhihu.com/equation?tex=+%5Cmin%5Climits_%7Ba%2Cs%7D%5CBigg%5B%5Cmin%5Climits_%7Bc_1%7D%5Csum%5Climits_%7Bx_i+%5Cin+D_1%7D%28y_i+-+c_1%29%5E2+%2B+%5Cmin%5Climits_%7Bc_2%7D%5Csum%5Climits_%7Bx_i+%5Cin+D_2%7D%28y_i+-+c_2%29%5E2%5CBigg%5D+%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=c_1) 为 ![[公式]](https://www.zhihu.com/equation?tex=D_1) 数据集的样本输出均值， ![[公式]](https://www.zhihu.com/equation?tex=c_2) 为 ![[公式]](https://www.zhihu.com/equation?tex=D_2) 数据集的样本输出均值。

**3.6.2 预测方式**

对于决策树建立后做预测的方式，上面讲到了 CART 分类树采用叶子节点里概率最大的类别作为当前节点的预测类别。而回归树输出不是类别，它采用的是用最终叶子的均值或者中位数来预测输出结果。

## 4. 总结

最后通过总结的方式对比下 ID3、C4.5 和 CART 三者之间的差异。

除了之前列出来的划分标准、剪枝策略、连续值确实值处理方式等之外，我再介绍一些其他差异：

- **划分标准的差异：**ID3 使用信息增益偏向特征值多的特征，C4.5 使用信息增益率克服信息增益的缺点，偏向于特征值小的特征，CART 使用基尼指数克服 C4.5 需要求 log 的巨大计算量，偏向于特征值较多的特征。
- **使用场景的差异：**ID3 和 C4.5 都只能用于分类问题，CART 可以用于分类和回归问题；ID3 和 C4.5 是多叉树，速度较慢，CART 是二叉树，计算速度很快；
- **样本数据的差异：**ID3 只能处理离散数据且缺失值敏感，C4.5 和 CART 可以处理连续性数据且有多种方式处理缺失值；从样本量考虑的话，小样本建议 C4.5、大样本建议 CART。C4.5 处理过程中需对数据集进行多次扫描排序，处理成本耗时较高，而 CART 本身是一种大样本的统计方法，小样本处理下泛化误差较大 ；
- **样本特征的差异：**ID3 和 C4.5 层级之间只使用一次特征，CART 可多次重复使用特征；
- **剪枝策略的差异：**ID3 没有剪枝策略，C4.5 是通过悲观剪枝策略来修正树的准确性，而 CART 是通过代价复杂度剪枝。





本文主要介绍基于集成学习的决策树，其主要通过不同学习框架生产基学习器，并综合所有基学习器的预测结果来改善单个基学习器的识别率和泛化性。

## 1. 集成学习

常见的集成学习框架有三种：Bagging，Boosting 和 Stacking。三种集成学习框架在基学习器的产生和综合结果的方式上会有些区别，我们先做些简单的介绍。

### 1.1 Bagging

Bagging 全称叫 **B**ootstrap **agg**regat**ing**，看到 Bootstrap 我们立刻想到著名的开源前端框架（抖个机灵，是 Bootstrap 抽样方法） ，每个基学习器都会对训练集进行有放回抽样得到子训练集，比较著名的采样法为 0.632 自助法。每个基学习器基于不同子训练集进行训练，并综合所有基学习器的预测值得到最终的预测结果。Bagging 常用的综合方法是投票法，票数最多的类别为预测类别。

![img](https://pic1.zhimg.com/80/v2-a0a3cb02f629f3db360fc68b4c2153c0_1440w.jpg)

### 1.2 Boosting

Boosting 训练过程为阶梯状，基模型的训练是有顺序的，每个基模型都会在前一个基模型学习的基础上进行学习，最终综合所有基模型的预测值产生最终的预测结果，用的比较多的综合方式为加权法。

![img](https://pic3.zhimg.com/80/v2-3aab53d50ab65e11ad3c9e3decf895c2_1440w.jpg)

### 1.3 Stacking

Stacking 是先用全部数据训练好基模型，然后每个基模型都对每个训练样本进行的预测，其预测值将作为训练样本的特征值，最终会得到新的训练样本，然后基于新的训练样本进行训练得到模型，然后得到最终预测结果。

![img](https://pic3.zhimg.com/80/v2-f6787a16c23950d129a7927269d5352a_1440w.jpg)

那么，为什么集成学习会好于单个学习器呢？原因可能有三：

1. 训练样本可能无法选择出最好的单个学习器，由于没法选择出最好的学习器，所以干脆结合起来一起用；
2. 假设能找到最好的学习器，但由于算法运算的限制无法找到最优解，只能找到次优解，采用集成学习可以弥补算法的不足；
3. 可能算法无法得到最优解，而集成学习能够得到近似解。比如说最优解是一条对角线，而单个决策树得到的结果只能是平行于坐标轴的，但是集成学习可以去拟合这条对角线。

## 2. 偏差与方差

上节介绍了集成学习的基本概念，这节我们主要介绍下如何从偏差和方差的角度来理解集成学习。

### 2.1 集成学习的偏差与方差

偏差（Bias）描述的是预测值和真实值之差；方差（Variance）描述的是预测值作为随机变量的离散程度。放一场很经典的图：



![img](https://pic2.zhimg.com/80/v2-60c942f91d33d9dedf9dd2c7d482af5d_1440w.jpg)

模型的偏差与方差

- **偏差：**描述样本拟合出的模型的预测结果的期望与样本真实结果的差距，要想偏差表现的好，就需要复杂化模型，增加模型的参数，但这样容易过拟合，过拟合对应上图的 High Variance，点会很分散。低偏差对应的点都打在靶心附近，所以喵的很准，但不一定很稳；
- **方差：**描述样本上训练出来的模型在测试集上的表现，要想方差表现的好，需要简化模型，减少模型的复杂度，但这样容易欠拟合，欠拟合对应上图 High Bias，点偏离中心。低方差对应就是点都打的很集中，但不一定是靶心附近，手很稳，但不一定瞄的准。

我们常说集成学习中的基模型是弱模型，通常来说弱模型是偏差高（在训练集上准确度低）方差小（防止过拟合能力强）的模型，**但并不是所有集成学习框架中的基模型都是弱模型**。**Bagging 和 Stacking 中的基模型为强模型（偏差低，方差高），而Boosting 中的基模型为弱模型（偏差高，方差低）**。

在 Bagging 和 Boosting 框架中，通过计算基模型的期望和方差我们可以得到模型整体的期望和方差。为了简化模型，我们假设基模型的期望为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) ，方差 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma+%5E+2) ，模型的权重为 ![[公式]](https://www.zhihu.com/equation?tex=r) ，两两模型间的相关系数 ![[公式]](https://www.zhihu.com/equation?tex=%5Crho) 相等。由于 Bagging 和 Boosting 的基模型都是线性组成的，那么有：

模型总体期望：
![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+++E%28F%29+%26%3D+E%28%5Csum_%7Bi%7D%5E%7Bm%7D%7Br_i+f_i%7D%29+++%5C%5C+%26%3D+%5Csum_%7Bi%7D%5E%7Bm%7Dr_i+E%28f_i%29+++%5Cend%7Balign%7D++%5C%5C)

模型总体方差（公式推导参考协方差的性质，协方差与方差的关系）：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+++Var%28F%29+%26%3D+Var%28%5Csum_%7Bi%7D%5E%7Bm%7D%7Br_i+f_i%7D%29+%5C%5C+++++++%26%3D+%5Csum_%7Bi%7D%5E%7Bm%7DVar%28r_if_i%29+%2B+%5Csum_%7Bi+%5Cneq+j%7D%5E%7Bm%7DCov%28r_i+f_i+%2C++r_j+f_j%29+++%5C%5C+%26%3D+%5Csum_%7Bi%7D%5E%7Bm%7D+%7Br_i%7D%5E2+Var%28f_i%29+%2B+%5Csum_%7Bi+%5Cneq+j%7D%5E%7Bm%7D%5Crho+r_i+r_j+%5Csqrt%7BVar%28f_i%29%7D+%5Csqrt%7BVar%28f_j%29%7D+%5C%5C+++%26%3D+mr%5E2%5Csigma%5E2+%2B+m%28m-1%29%5Crho+r%5E2+%5Csigma%5E2%5C%5C+++%26%3D+m+r%5E2+%5Csigma%5E2++%281-%5Crho%29+%2B++m%5E2+r%5E2+%5Csigma%5E2+%5Crho+%5Cend%7Balign%7D++%5C%5C)

模型的准确度可由偏差和方差共同决定：

![[公式]](https://www.zhihu.com/equation?tex=Error+%3D+bias%5E2+%2B+var+%2B+%5Cxi+%5C%5C)

### 2.2 Bagging 的偏差与方差

对于 Bagging 来说，每个基模型的权重等于 1/m 且期望近似相等，故我们可以得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+++E%28F%29+%26+%3D+%5Csum_%7Bi%7D%5E%7Bm%7Dr_i+E%28f_i%29+++%5C%5C++++++%26%3D+m+%5Cfrac%7B1%7D%7Bm%7D+%5Cmu+%5C%5C+++++%26%3D+%5Cmu++%5C%5C+++Var%28F%29+%26%3D++m+r%5E2+%5Csigma%5E2+%281-%5Crho%29+%2B+m%5E2+r%5E2+%5Csigma%5E2+%5Crho+%5C%5C+++++%26%3D+m+%5Cfrac%7B1%7D%7Bm%5E2%7D+%5Csigma%5E2+%281-%5Crho%29+%2B+m%5E2+%5Cfrac%7B1%7D%7Bm%5E2%7D+%5Csigma%5E2+%5Crho++%5C%5C+++++%26%3D+%5Cfrac%7B%5Csigma%5E2%281+-+%5Crho%29%7D%7Bm%7D++%2B+%5Csigma%5E2+%5Crho++%5Cend%7Balign%7D++%5C%5C)

通过上式我们可以看到：

- **整体模型的期望等于基模型的期望，这也就意味着整体模型的偏差和基模型的偏差近似。**
- **整体模型的方差小于等于基模型的方差，当且仅当相关性为 1 时取等号，随着基模型数量增多，整体模型的方差减少，从而防止过拟合的能力增强，模型的准确度得到提高。**但是，模型的准确度一定会无限逼近于 1 吗？并不一定，当基模型数增加到一定程度时，方差公式第一项的改变对整体方差的作用很小，防止过拟合的能力达到极限，这便是准确度的极限了。

在此我们知道了为什么 Bagging 中的基模型一定要为强模型，如果 Bagging 使用弱模型则会导致整体模型的偏差提高，而准确度降低。

Random Forest 是经典的基于 Bagging 框架的模型，并在此基础上通过引入特征采样和样本采样来降低基模型间的相关性，在公式中显著降低方差公式中的第二项，略微升高第一项，从而使得整体降低模型整体方差。

### 2.3 Boosting 的偏差与方差

对于 Boosting 来说，由于基模型共用同一套训练集，所以基模型间具有强相关性，故模型间的相关系数近似等于 1，针对 Boosting 化简公式为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D++E%28F%29+%26+%3D+%5Csum_%7Bi%7D%5E%7Bm%7Dr_i+E%28f_i%29+%5C%5C++Var%28F%29+%26%3D+m+r%5E2+%5Csigma%5E2++%281-%5Crho%29+%2B++m%5E2+r%5E2+%5Csigma%5E2+%5Crho+%5C%5C+++%26%3D+m+%5Cfrac%7B1%7D%7Bm%5E2%7D+%5Csigma%5E2+%281-1%29+%2B+m%5E2+%5Cfrac%7B1%7D%7Bm%5E2%7D+%5Csigma%5E2+1++%5C%5C%26%3D++%5Csigma%5E2+++%5Cend%7Balign%7D++%5C%5C)

通过观察整体方差的表达式我们容易发现：

- 整体模型的方差等于基模型的方差，如果基模型不是弱模型，其方差相对较大，这将导致整体模型的方差很大，即无法达到防止过拟合的效果。因此，Boosting 框架中的基模型必须为弱模型。
- 此外 Boosting 框架中采用基于贪心策略的前向加法，整体模型的期望由基模型的期望累加而成，所以随着基模型数的增多，整体模型的期望值增加，整体模型的准确度提高。

基于 Boosting 框架的 Gradient Boosting Decision Tree 模型中基模型也为树模型，同 Random Forrest，我们也可以对特征进行随机抽样来使基模型间的相关性降低，从而达到减少方差的效果。

### 2.4 小结

- 我们可以使用模型的偏差和方差来近似描述模型的准确度；
- 对于 Bagging 来说，整体模型的偏差与基模型近似，而随着模型的增加可以降低整体模型的方差，故其基模型需要为强模型；
- 对于 Boosting 来说，整体模型的方差近似等于基模型的方差，而整体模型的偏差由基模型累加而成，故基模型需要为弱模型。

️那么这里有一个小小的疑问，Bagging 和 Boosting 到底用的是什么模型呢？

## 3. Random Forest

Random Forest（随机森林），用随机的方式建立一个森林。RF 算法由很多决策树组成，每一棵决策树之间没有关联。建立完森林后，当有新样本进入时，每棵决策树都会分别进行判断，然后基于投票法给出分类结果。

### 3.1 思想

Random Forest（随机森林）是 Bagging 的扩展变体，它在以决策树为基学习器构建 Bagging 集成的基础上，进一步在决策树的训练过程中引入了随机特征选择，因此可以概括 RF 包括四个部分：

1. 随机选择样本（放回抽样）；
2. 随机选择特征；
3. 构建决策树；
4. 随机森林投票（平均）。

随机选择样本和 Bagging 相同，采用的是 Bootstrap 自助采样法；**随机选择特征是指在每个节点在分裂过程中都是随机选择特征的**（区别与每棵树随机选择一批特征）。

这种随机性导致随机森林的偏差会有稍微的增加（相比于单棵不随机树），但是由于随机森林的“平均”特性，会使得它的方差减小，而且方差的减小补偿了偏差的增大，因此总体而言是更好的模型。

随机采样由于引入了两种采样方法保证了随机性，所以每棵树都是最大可能的进行生长就算不剪枝也不会出现过拟合。

### 3.2 优缺点

**优点**

1. 在数据集上表现良好，相对于其他算法有较大的优势
2. 易于并行化，在大数据集上有很大的优势；
3. 能够处理高维度数据，不用做特征选择。

## 4 Adaboost

AdaBoost（Adaptive Boosting，自适应增强），其自适应在于：**前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。**

### 4.1 思想

Adaboost 迭代算法有三步：

1. 初始化训练样本的权值分布，每个样本具有相同权重；
2. 训练弱分类器，如果样本分类正确，则在构造下一个训练集中，它的权值就会被降低；反之提高。用更新过的样本集去训练下一个分类器；
3. 将所有弱分类组合成强分类器，各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，降低分类误差率大的弱分类器的权重。

### 4.2 细节

**4.2.1 损失函数**

Adaboost 模型是加法模型，学习算法为前向分步学习算法，损失函数为指数函数的分类问题。

**加法模型**：最终的强分类器是由若干个弱分类器加权平均得到的。

**前向分布学习算法**：算法是通过一轮轮的弱学习器学习，利用前一个弱学习器的结果来更新后一个弱学习器的训练集权重。第 k 轮的强学习器为：

![[公式]](https://www.zhihu.com/equation?tex=+F_%7Bk%7D%28x%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%5Calpha_i+f_i%28x%29%3DF_%7Bk-1%7D%28x%29%2B%5Calpha_%7Bk%7Df_k%28x%29+%5C%5C)

定义损失函数为 n 个样本的指数损失函数：

![[公式]](https://www.zhihu.com/equation?tex=L%28y%2CF%29+%3D+%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bn%7Dexp%28-y_iF_%7Bk%7D%28x_i%29%29++%5C%5C)

利用前向分布学习算法的关系可以得到：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D++L%28y%2C+F%29+%26%3D+%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bm%7Dexp%5B%28-y_i%29+%28F_%7Bk-1%7D%28x_i%29+%2B+%5Calpha_k+f_k%28x_i%29%29%5D++%5C%5C+%26%3D++%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bm%7Dexp%5B-y_i+F_%7Bk-1%7D%28x_i%29+-y_i++%5Calpha_k+f_k%28x_i%29%5D+%5C%5C+%26%3D++%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bm%7Dexp%5B-y_i+F_%7Bk-1%7D%28x_i%29+%5D+exp%5B-y_i++%5Calpha_k+f_k%28x_i%29%5D+++%5Cend%7Balign%7D++%5C%5C)

因为 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bk-1%7D%28x%29) 已知，所以令 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bk%2Ci%7D+%3D+exp%28-y_iF_%7Bk-1%7D%28x_i%29%29) ，随着每一轮迭代而将这个式子带入损失函数，损失函数转化为：

![[公式]](https://www.zhihu.com/equation?tex=L%28y%2C+F%28x%29%29+%3D%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bm%7Dw_%7Bk%2Ci%7Dexp%5B-y_i%5Calpha_k+f_k%28x_i%29%5D+%5C%5C)

我们求 ![[公式]](https://www.zhihu.com/equation?tex=f_k%28x%29) ，可以得到：

![[公式]](https://www.zhihu.com/equation?tex=f_k%28x%29+%3Dargmin%5C%3B+%5Csum_%5Climits%7Bi%3D1%7D%5E%7Bm%7Dw_%7Bk%2Ci%7DI%28y_i+%5Cneq+f_k%28x_i%29%29+%5C%5C)

将 ![[公式]](https://www.zhihu.com/equation?tex=f_k%28x%29) 带入损失函数，并对 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 求导，使其等于 0，则就得到了：

![[公式]](https://www.zhihu.com/equation?tex=+%5Calpha_k+%3D+%5Cfrac%7B1%7D%7B2%7Dlog%5Cfrac%7B1-e_k%7D%7Be_k%7D++%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=e_k) 即为我们前面的分类误差率。

![[公式]](https://www.zhihu.com/equation?tex=+e_k+%3D+%5Cfrac%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bm%7Dw_%7Bki%7D%5E%7B%E2%80%99%7DI%28y_i+%5Cneq+f_k%28x_i%29%29%7D%7B%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bm%7Dw_%7Bki%7D%5E%7B%E2%80%99%7D%7D+%3D+%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bm%7Dw_%7Bki%7DI%28y_i+%5Cneq+f_k%28x_i%29%29+%5C%5C)

最后看样本权重的更新。利用 ![[公式]](https://www.zhihu.com/equation?tex=F_%7Bk%7D%28x%29+%3D+F_%7Bk-1%7D%28x%29+%2B+%5Calpha_kf_k%28x%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bk%2B1%2Ci%7D%3Dw_%7Bk%2Ci%7Dexp%5B-y_i%5Calpha_kf_k%28x%2Ci%29%5D) ，即可得：

![[公式]](https://www.zhihu.com/equation?tex=w_%7Bk%2B1%2Ci%7D+%3D+w_%7Bki%7Dexp%5B-y_i%5Calpha_kf_k%28x_i%29%5D+%5C%5C)

这样就得到了样本权重更新公式。

**4.2.2 正则化**

为了防止 Adaboost 过拟合，我们通常也会加入正则化项，这个正则化项我们通常称为步长（learning rate）。对于前面的弱学习器的迭代

![[公式]](https://www.zhihu.com/equation?tex=F_%7Bk%7D%28x%29+%3D+F_%7Bk-1%7D%28x%29+%2B+%5Calpha_kf_k%28x%29+%5C%5C)

加上正则化项 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu+) 我们有：

![[公式]](https://www.zhihu.com/equation?tex=F_%7Bk%7D%28x%29+%3D+F_%7Bk-1%7D%28x%29+%2B+%5Cmu%5Calpha_kf_k%28x%29++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) 的取值范围为 ![[公式]](https://www.zhihu.com/equation?tex=0%3C%5Cmu%5Cleq1) 。对于同样的训练集学习效果，较小的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) 意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。

### 4.3 优缺点

**4.3.1 优点**

1. 分类精度高；
2. 可以用各种回归分类模型来构建弱学习器，非常灵活；
3. 不容易发生过拟合。

**4.3.2 缺点**

1. 对异常点敏感，异常点会获得较高权重。

## 5. GBDT

GBDT（Gradient Boosting Decision Tree）是一种迭代的决策树算法，该算法由多棵决策树组成，从名字中我们可以看出来它是属于 Boosting 策略。GBDT 是被公认的泛化能力较强的算法。

### 5.1 思想

GBDT 由三个概念组成：Regression Decision Tree（即 DT）、Gradient Boosting（即 GB），和 Shrinkage（一个重要演变）

**5.1.1 回归树（Regression Decision Tree）**

如果认为 GBDT 由很多分类树那就大错特错了（虽然调整后也可以分类）。对于分类树而言，其值加减无意义（如性别），而对于回归树而言，其值加减才是有意义的（如说年龄）。GBDT 的核心在于累加所有树的结果作为最终结果，所以 GBDT 中的树都是回归树，不是分类树，这一点相当重要。

回归树在分枝时会穷举每一个特征的每个阈值以找到最好的分割点，衡量标准是最小化均方误差。

**5.1.2 梯度迭代（Gradient Boosting）**

上面说到 GBDT 的核心在于累加所有树的结果作为最终结果，GBDT 的每一棵树都是以之前树得到的残差来更新目标值，这样每一棵树的值加起来即为 GBDT 的预测值。

模型的预测值可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=+F_k%28x%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bk%7Df_%7Bi%7D%28x%29+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=f_%7Bi%7D%28x%29+) 为基模型与其权重的乘积，模型的训练目标是使预测值 ![[公式]](https://www.zhihu.com/equation?tex=F_k%28x%29) 逼近真实值 y，也就是说要让每个基模型的预测值逼近各自要预测的部分真实值。由于要同时考虑所有基模型，导致了整体模型的训练变成了一个非常复杂的问题。所以研究者们想到了一个贪心的解决手段：每次只训练一个基模型。那么，现在改写整体模型为迭代式：

![[公式]](https://www.zhihu.com/equation?tex=F_k%28x%29+%3D+F_%7Bk-1%7D%28x%29%2Bf_%7Bk%7D%28x%29%5C%5C)

这样一来，每一轮迭代中，只要集中解决一个基模型的训练问题：使 ![[公式]](https://www.zhihu.com/equation?tex=F_k%28x%29) 逼近真实值 ![[公式]](https://www.zhihu.com/equation?tex=y) 。

举个例子：比如说 A 用户年龄 20 岁，第一棵树预测 12 岁，那么残差就是 8，第二棵树用 8 来学习，假设其预测为 5，那么其残差即为 3，如此继续学习即可。

那么 Gradient 从何体现？其实很简单，其**残差其实是最小均方损失函数关于预测值的反向梯度(划重点)**：

![[公式]](https://www.zhihu.com/equation?tex=+-%5Cfrac%7B%5Cpartial+%28%5Cfrac%7B1%7D%7B2%7D%28y-F_%7Bk%7D%28x%29%29%5E2%29%7D%7B%5Cpartial+F_k%28x%29%7D+%3D+y-F_%7Bk%7D%28x%29++%5C%5C)

也就是说，预测值和实际值的残差与损失函数的负梯度相同。

但要注意，基于残差 GBDT 容易对异常值敏感，举例：



![img](https://pic2.zhimg.com/80/v2-52f8fe9d1990c31335bf26c0c85d7ad5_1440w.jpg)



很明显后续的模型会对第 4 个值关注过多，这不是一种好的现象，所以一般回归类的损失函数会用**绝对损失或者 Huber 损失函数**来代替平方损失函数。



![img](https://pic3.zhimg.com/80/v2-c657e78a5a9e3646dc493a3f69556c8a_1440w.jpg)



GBDT 的 Boosting 不同于 Adaboost 的 Boosting，**GBDT 的每一步残差计算其实变相地增大了被分错样本的权重，而对与分对样本的权重趋于 0**，这样后面的树就能专注于那些被分错的样本。

**5.1.3 缩减（Shrinkage）**

Shrinkage 的思想认为，每走一小步逐渐逼近结果的效果要比每次迈一大步很快逼近结果的方式更容易避免过拟合。即它并不是完全信任每一棵残差树。

![[公式]](https://www.zhihu.com/equation?tex=F_i%28x%29%3DF_%7Bi-1%7D%28x%29%2B%5Cmu+f_i%28x%29+%5Cquad+%280%3C%5Cmu+%5Cleq+1%29++%5C%5C)

Shrinkage 不直接用残差修复误差，而是只修复一点点，把大步切成小步。本质上 Shrinkage 为每棵树设置了一个 weight，累加时要乘以这个 weight，当 weight 降低时，基模型数会配合增大。

### 5.2 优缺点

**5.2.1 优点**

1. 可以自动进行特征组合，拟合非线性数据；
2. 可以灵活处理各种类型的数据。

**5.2.2 缺点**

1. 对异常点敏感。

### 5.3 与 Adaboost 的对比

**5.3.1 相同：**

1. 都是 Boosting 家族成员，使用弱分类器；
2. 都使用前向分布算法；

**5.3.2 不同：**

1. **迭代思路不同**：Adaboost 是通过提升错分数据点的权重来弥补模型的不足（利用错分样本），而 GBDT 是通过算梯度来弥补模型的不足（利用残差）；
2. **损失函数不同**：AdaBoost 采用的是指数损失，GBDT 使用的是绝对损失或者 Huber 损失函数；



本文是决策树的第三篇，主要介绍基于 Boosting 框架的主流集成算法，包括 XGBoost 和 LightGBM。

不知道为什么知乎文章封面的照片会显示不全，在这里补上完整的思维导图：

![img](https://pic2.zhimg.com/80/v2-358e4bfce928d0460bd5e8b4cab8f715_1440w.jpg)

## 1. XGBoost

XGBoost 是大规模并行 boosting tree 的工具，它是目前最快最好的开源 boosting tree 工具包，比常见的工具包快 10 倍以上。Xgboost 和 GBDT 两者都是 boosting 方法，除了工程实现、解决问题上的一些差异外，最大的不同就是目标函数的定义。故本文将从数学原理和工程实现上进行介绍，并在最后介绍下 Xgboost 的优点。

### 1.1 数学原理

**1.1.1 目标函数**

我们知道 XGBoost 是由 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型组成的一个加法运算式：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%3D%5Csum_%7Bt%3D1%7D%5E%7Bk%7D%5C+f_t%28x_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=f_k) 为第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 为第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本的预测值。

损失函数可由预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 进行表示：

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bi%3D1%7D%5En+l%28+y_i%2C+%5Chat%7By%7D_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 为样本数量。

我们知道模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要简单的模型，所以目标函数由模型的损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L) 与抑制模型复杂度的正则项 ![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 组成，所以我们有：

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D%5Csum_%7Bi%3D1%7D%5En+l%28%5Chat%7By%7D_i%2C+y_i%29+%2B+%5Csum_%7Bt%3D1%7D%5Ek+%5COmega%28f_t%29+%5C%5C+)

![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 为模型的正则项，由于 XGBoost 支持决策树也支持线性模型，所以这里再不展开描述。

我们知道 boosting 模型是前向加法，以第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步的模型为例，模型对第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 的预测为：

![[公式]](https://www.zhihu.com/equation?tex=++%5Chat%7By%7D_i%5Et%3D+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29++%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 由第 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 步的模型给出的预测值，是已知常数，![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 是我们这次需要加入的新模型的预测值，此时，目标函数就可以写成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Obj%5E%7B%28t%29%7D+%26%3D+%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C+%5Chat%7By%7D_i%5Et%29+%2B+%5Csum_%7Bi%3D1%7D%5Et%5COmega%28f_i%29+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5En+l%5Cleft%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29+%5Cright%29+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29++%5Cend%7Balign%7D+%5C%5C)

求此时最优化目标函数，就相当于求解 ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 。

> 泰勒公式是将一个在 ![[公式]](https://www.zhihu.com/equation?tex=x%3Dx_0) 处具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数的函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 利用关于 ![[公式]](https://www.zhihu.com/equation?tex=x-x_0) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 次多项式来逼近函数的方法，若函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 在包含 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 的某个闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数，且在开区间 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%29) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n%2B1) 阶导数，则对闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上任意一点 ![[公式]](https://www.zhihu.com/equation?tex=x) 有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+f%28x%29%3D%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%5Cfrac%7Bf%5E%7B%28i%29%7D%28x_0%29%7D%7Bi%21%7D%28x-x_0%29%5E+i%2BR_n%28x%29+) ，其中的多项式称为函数在 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 处的泰勒展开式， ![[公式]](https://www.zhihu.com/equation?tex=R_n%28x%29) 是泰勒公式的余项且是 ![[公式]](https://www.zhihu.com/equation?tex=%28x%E2%88%92x_0%29%5En) 的高阶无穷小。

根据泰勒公式我们把函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29) 在点 ![[公式]](https://www.zhihu.com/equation?tex=x) 处进行泰勒的二阶展开，可得到如下等式：

![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29+%5Capprox+f%28x%29+%2B+f%27%28x%29%5CDelta+x+%2B+%5Cfrac12+f%27%27%28x%29%5CDelta+x%5E2++%5C%5C)

我们把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 视为 ![[公式]](https://www.zhihu.com/equation?tex=x) ， ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 视为 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta+x) ，故可以将目标函数写为：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29+%2B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bi%7D) 为损失函数的一阶导， ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D) 为损失函数的二阶导，**注意这里的导是对 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 求导**。

我们以平方损失函数为例：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5En+%5Cleft%28y_i+-+%28%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29%29+%5Cright%29%5E2++%5C%5C)

则：

![[公式]](https://www.zhihu.com/equation?tex=++%5Cbegin%7Balign%7D++++++g_i+%26%3D+%5Cfrac%7B%5Cpartial+%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%5Cpartial+%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29+%5C%5C++++++h_i+%26%3D%5Cfrac%7B%5Cpartial%5E2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2++++%5Cend%7Balign%7D++%5C%5C)

由于在第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步时 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 其实是一个已知的值，所以 ![[公式]](https://www.zhihu.com/equation?tex=l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29) 是一个常数，其对函数的优化不会产生影响，因此目标函数可以写成：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

所以我们只需要求出每一步损失函数的一阶导和二阶导的值（由于前一步的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7Bt-1%7D) 是已知的，所以这两个值就是常数），然后最优化目标函数，就可以得到每一步的 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，最后根据加法模型得到一个整体模型。

**1.1.2 基于决策树的目标函数**

我们知道 Xgboost 的基模型**不仅支持决策树，还支持线性模型**，这里我们主要介绍基于决策树的目标函数。

我们可以将决策树定义为 ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x%29%3Dw_%7Bq%28x%29%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=x) 为某一样本，这里的 ![[公式]](https://www.zhihu.com/equation?tex=q%28x%29) 代表了该样本在哪个叶子结点上，而 ![[公式]](https://www.zhihu.com/equation?tex=w_q) 则代表了叶子结点取值 ![[公式]](https://www.zhihu.com/equation?tex=w) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bq%28x%29%7D) 就代表了每个样本的取值 ![[公式]](https://www.zhihu.com/equation?tex=w) （即预测值)。

决策树的复杂度可由叶子数 ![[公式]](https://www.zhihu.com/equation?tex=T) 组成，叶子节点越少模型越简单，此外叶子节点也不应该含有过高的权重 ![[公式]](https://www.zhihu.com/equation?tex=w) （类比 LR 的每个变量的权重)，所以目标函数的正则项可以定义为：

![[公式]](https://www.zhihu.com/equation?tex=%5COmega%28f_t%29%3D%5Cgamma+T+%2B+%5Cfrac12+%5Clambda+%5Csum_%7Bj%3D1%7D%5ET+w_j%5E2+%5C%5C)

即决策树模型的复杂度由生成的所有决策树的叶子节点数量，和所有节点权重所组成的向量的 ![[公式]](https://www.zhihu.com/equation?tex=L_2) 范式共同决定。

![img](https://pic1.zhimg.com/80/v2-e0ab9287990a6098e4cdbc5a8cff4150_1440w.jpg)

这张图给出了基于决策树的 XGBoost 的正则项的求解方式。

我们设 ![[公式]](https://www.zhihu.com/equation?tex=I_j%3D+%5C%7B+i+%5Cvert+q%28x_i%29%3Dj+%5C%7D) 为第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个叶子节点的样本集合，故我们的目标函数可以写成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Obj%5E%7B%28t%29%7D+%26%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5COmega%28f_t%29+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_iw_%7Bq%28x_i%29%7D+%2B+%5Cfrac12h_iw_%7Bq%28x_i%29%7D%5E2+%5Cright%5D+%2B+%5Cgamma+T+%2B+%5Cfrac12+%5Clambda+%5Csum_%7Bj%3D1%7D%5ETw_j%5E2+%5C%5C++++%26%3D+%5Csum_%7Bj%3D1%7D%5ET+%5Cleft%5B%28%5Csum_%7Bi+%5Cin+I_j%7Dg_i%29w_j+%2B+%5Cfrac12%28%5Csum_%7Bi+%5Cin+I_j%7Dh_i+%2B+%5Clambda%29w_j%5E2+%5Cright%5D+%2B+%5Cgamma+T+%5Cend%7Balign%7D+%5C%5C)

第二步到第三步可能看的不是特别明白，这边做些解释：第二步是遍历所有的样本后求每个样本的损失函数，但样本最终会落在叶子节点上，所以我们也可以遍历叶子节点，然后获取叶子节点上的样本集合，最后在求损失函数。即我们之前样本的集合，现在都改写成叶子结点的集合，由于一个叶子结点有多个样本存在，因此才有了 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi+%5Cin+I_j%7Dg_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi+%5Cin+I_j%7Dh_i) 这两项， ![[公式]](https://www.zhihu.com/equation?tex=w_j) 为第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个叶子节点取值。

为简化表达式，我们定义 ![[公式]](https://www.zhihu.com/equation?tex=G_j%3D%5Csum_%7Bi+%5Cin+I_j%7Dg_i) ， ![[公式]](https://www.zhihu.com/equation?tex=H_j%3D%5Csum_%7Bi+%5Cin+I_j%7Dh_i) ，则目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%3D+%5Csum_%7Bj%3D1%7D%5ET+%5Cleft%5BG_jw_j+%2B+%5Cfrac12%28H_j+%2B+%5Clambda%29w_j%5E2+%5Cright%5D+%2B+%5Cgamma+T+%5C%5C)

这里我们要注意 ![[公式]](https://www.zhihu.com/equation?tex=G_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=H_j) 是前 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 步得到的结果，其值已知可视为常数，只有最后一棵树的叶子节点 ![[公式]](https://www.zhihu.com/equation?tex=w_j) 不确定，那么将目标函数对 ![[公式]](https://www.zhihu.com/equation?tex=w_j) 求一阶导，并令其等于 ![[公式]](https://www.zhihu.com/equation?tex=0) ，则可以求得叶子结点 ![[公式]](https://www.zhihu.com/equation?tex=j) 对应的权值：

![[公式]](https://www.zhihu.com/equation?tex=w_j%5E%2A%3D-%5Cfrac%7BG_j%7D%7BH_j%2B%5Clambda%7D++%5C%5C)

所以目标函数可以化简为：

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D+-%5Cfrac12+%5Csum_%7Bj%3D1%7D%5ET+%5Cfrac%7BG_j%5E2%7D%7BH_j%2B%5Clambda%7D+%2B+%5Cgamma+T+%5C%5C)

![img](https://pic2.zhimg.com/80/v2-f6db7af6c1e683192cb0ccf48eafaf99_1440w.jpg)

上图给出目标函数计算的例子，求每个节点每个样本的一阶导数 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 和二阶导数 ![[公式]](https://www.zhihu.com/equation?tex=h_i) ，然后针对每个节点对所含样本求和得到的 ![[公式]](https://www.zhihu.com/equation?tex=G_j) 和 ![[公式]](https://www.zhihu.com/equation?tex=H_j) ，最后遍历决策树的节点即可得到目标函数。

**1.1.3 最优切分点划分算法**

在决策树的生长过程中，一个非常关键的问题是如何找到叶子的节点的最优切分点，Xgboost 支持两种分裂节点的方法——贪心算法和近似算法。

**1）贪心算法**

1. 从深度为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的树开始，对每个叶节点枚举所有的可用特征；
2. 针对每个特征，把属于该节点的训练样本根据该特征值进行升序排列，通过线性扫描的方式来决定该特征的最佳分裂点，并记录该特征的分裂收益；
3. 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，在该节点上分裂出左右两个新的叶节点，并为每个新节点关联对应的样本集
4. 回到第 1 步，递归执行到满足特定条件为止

那么如何计算每个特征的分裂收益呢？

假设我们在某一节点完成特征分裂，则分列前的目标函数可以写为：

![[公式]](https://www.zhihu.com/equation?tex=Obj_%7B1%7D+%3D-%5Cfrac12+%5B%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5D+%2B+%5Cgamma++%5C%5C)

分裂后的目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=Obj_2+%3D++-%5Cfrac12+%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D%5D+%2B2%5Cgamma+%5C%5C)

则对于目标函数来说，分裂后的收益为：

![[公式]](https://www.zhihu.com/equation?tex=Gain%3D%5Cfrac12+%5Cleft%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D+-+%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5Cright%5D+-+%5Cgamma+%5C%5C)

注意该特征收益也可作为特征重要性输出的重要依据。

对于每次分裂，我们都需要枚举所有特征可能的分割方案，如何高效地枚举所有的分割呢？

我假设我们要枚举所有 ![[公式]](https://www.zhihu.com/equation?tex=x+%3C+a) 这样的条件，对于某个特定的分割点 ![[公式]](https://www.zhihu.com/equation?tex=a) 我们要计算 ![[公式]](https://www.zhihu.com/equation?tex=a) 左边和右边的导数和。

![img](https://pic2.zhimg.com/80/v2-79a82ed4f272bdf2d1cb77a514e40075_1440w.jpg)

我们可以发现对于所有的分裂点 ![[公式]](https://www.zhihu.com/equation?tex=a) ，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和 ![[公式]](https://www.zhihu.com/equation?tex=G_L) 和 ![[公式]](https://www.zhihu.com/equation?tex=G_R) 。然后用上面的公式计算每个分割方案的分数就可以了。

观察分裂后的收益，我们会发现节点划分不一定会使得结果变好，因为我们有一个引入新叶子的惩罚项，也就是说引入的分割带来的增益如果小于一个阀值的时候，我们可以剪掉这个分割。

**2）近似算法**

贪婪算法可以的到最优解，但当数据量太大时则无法读入内存进行计算，近似算法主要针对贪婪算法这一缺点给出了近似最优解。

对于每个特征，只考察分位点可以减少计算复杂度。

该算法会首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的桶中，然后聚合统计信息找到所有区间的最佳分裂点。

在提出候选切分点时有两种策略：

- Global：学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
- Local：每次分裂前将重新提出候选切分点。

直观上来看，Local 策略需要更多的计算步骤，而 Global 策略因为节点没有划分所以需要更多的候选点。

下图给出不同种分裂策略的 AUC 变换曲线，横坐标为迭代次数，纵坐标为测试集 AUC，eps 为近似算法的精度，其倒数为桶的数量。

![img](https://pic3.zhimg.com/80/v2-1da040923ad9beaf222a2dd60a8f3752_1440w.jpg)

我们可以看到 Global 策略在候选点数多时（eps 小）可以和 Local 策略在候选点少时（eps 大）具有相似的精度。此外我们还发现，在 eps 取值合理的情况下，分位数策略可以获得与贪婪算法相同的精度。

![img](https://pic1.zhimg.com/80/v2-161382c979557b8bae1563a459cd1ed4_1440w.jpg)

- **第一个 for 循环：**对特征 k 根据该特征分布的分位数找到切割点的候选集合 ![[公式]](https://www.zhihu.com/equation?tex=S_k%3D%5C%7Bs_%7Bk1%7D%2Cs_%7Bk2%7D%2C...%2Cs_%7Bkl%7D+%5C%7D) 。XGBoost 支持 Global 策略和 Local 策略。
- **第二个 for 循环：**针对每个特征的候选集合，将样本映射到由该特征对应的候选点集构成的分桶区间中，即 ![[公式]](https://www.zhihu.com/equation?tex=%7Bs_%7Bk%2Cv%7D%E2%89%A5x_%7Bjk%7D%3Es_%7Bk%2Cv%E2%88%921%7D%7D) ，对每个桶统计 ![[公式]](https://www.zhihu.com/equation?tex=G%2CH+) 值，最后在这些统计量上寻找最佳分裂点。

下图给出近似算法的具体例子，以三分位为例：

![img](https://pic2.zhimg.com/80/v2-5d1dd1673419599094bf44dd4b533ba9_1440w.jpg)

根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的 ![[公式]](https://www.zhihu.com/equation?tex=G%2CH) 值，最终求解节点划分的增益。

**1.1.4 加权分位数缩略图**

事实上， XGBoost 不是简单地按照样本个数进行分位，而是以二阶导数值 ![[公式]](https://www.zhihu.com/equation?tex=h_i+) 作为样本的权重进行划分，如下：

![img](https://pic4.zhimg.com/80/v2-5f16246289eaa2a3ae72f971db198457_1440w.jpg)

那么问题来了：为什么要用 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 进行样本加权？

我们知道模型的目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

我们稍作整理，便可以看出 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 有对 loss 加权的作用。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D++Obj%5E%7B%28t%29%7D+%26+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5B+g_i+f_t%28x_i%29+%2B+%5Cfrac%7B1%7D%7B2%7Dh_i+f_t%5E2%28x_i%29+%5Ccolor%7Bred%7D%7B%2B+%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7Bg_i%5E2%7D%7Bh_i%7D%7D%5D%2B%5COmega%28f_t%29+%5Ccolor%7Bred%7D%7B%2B+C%7D+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Ccolor%7Bred%7D%7B%5Cfrac%7B1%7D%7B2%7Dh_i%7D+%5Cleft%5B+f_t%28x_i%29+-+%5Cleft%28+-%5Cfrac%7Bg_i%7D%7Bh_i%7D+%5Cright%29+%5Cright%5D%5E2+%2B+%5COmega%28f_t%29+%2B+C+%5Cend%7Balign%7D+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7Bg_i%5E2%7D%7Bh_i%7D) 与 ![[公式]](https://www.zhihu.com/equation?tex=C) 皆为常数。我们可以看到 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 就是平方损失函数中样本的权重。

对于样本权值相同的数据集来说，找到候选分位点已经有了解决方案（GK 算法），但是当样本权值不一样时，该如何找到候选分位点呢？（作者给出了一个 Weighted Quantile Sketch 算法，这里将不做介绍。）

**1.1.5 稀疏感知算法**

在决策树的第一篇文章中我们介绍 CART 树在应对数据缺失时的分裂策略，XGBoost 也给出了其解决方案。

XGBoost 在构建树的节点过程中只考虑非缺失值的数据遍历，而为每个节点增加了一个缺省方向，当样本相应的特征值缺失时，可以被归类到缺省方向上，最优的缺省方向可以从数据中学到。至于如何学到缺省值的分支，其实很简单，分别枚举特征缺省的样本归为左右分支后的增益，选择增益最大的枚举项即为最优缺省方向。

在构建树的过程中需要枚举特征缺失的样本，乍一看该算法的计算量增加了一倍，但其实该算法在构建树的过程中只考虑了特征未缺失的样本遍历，而特征值缺失的样本无需遍历只需直接分配到左右节点，故算法所需遍历的样本量减少，下图可以看到稀疏感知算法比 basic 算法速度块了超过 50 倍。

![img](https://pic1.zhimg.com/80/v2-e065bea4b424ea2d13b25ed2e7004aa8_1440w.jpg)

### 1.2 工程实现

**1.2.1 块结构设计**

我们知道，决策树的学习最耗时的一个步骤就是在每次寻找最佳分裂点是都需要对特征的值进行排序。而 XGBoost 在训练之前对根据特征对数据进行了排序，然后保存到块结构中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

- 每一个块结构包括一个或多个已经排序好的特征；
- 缺失特征值将不进行排序；
- 每个特征会存储指向样本梯度统计值的索引，方便计算一阶导和二阶导数值；

![img](https://pic2.zhimg.com/80/v2-00c089b3439b2cdab85116d7cea511c5_1440w.jpg)

这种块结构存储的特征之间相互独立，方便计算机进行并行计算。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 Xgboost 能够实现分布式或者多线程计算的原因。

**1.2.2 缓存访问优化算法**

块结构的设计可以减少节点分裂时的计算量，但特征值通过索引访问样本梯度统计值的设计会导致访问操作的内存空间不连续，这样会造成缓存命中率低，从而影响到算法的效率。

为了解决缓存命中率低的问题，XGBoost 提出了缓存访问优化算法：为每个线程分配一个连续的缓存区，将需要的梯度信息存放在缓冲区中，这样就是实现了非连续空间到连续空间的转换，提高了算法效率。

此外适当调整块大小，也可以有助于缓存优化。

**1.2.3 “核外”块计算**

当数据量过大时无法将数据全部加载到内存中，只能先将无法加载到内存中的数据暂存到硬盘中，直到需要时再进行加载计算，而这种操作必然涉及到因内存与硬盘速度不同而造成的资源浪费和性能瓶颈。为了解决这个问题，XGBoost 独立一个线程专门用于从硬盘读入数据，以实现处理数据和读入数据同时进行。

此外，XGBoost 还用了两种方法来降低硬盘读写的开销：

- **块压缩：**对 Block 进行按列压缩，并在读取时进行解压；
- **块拆分：**将每个块存储到不同的磁盘中，从多个磁盘读取可以增加吞吐量。

### 1.3 优缺点

**1.3.1 优点**

1. **精度更高：**GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
2. **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，（使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题））。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
3. **正则化：**XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合；
4. **Shrinkage（缩减）：**相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；
5. **列抽样：**XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算；
6. **缺失值处理：**XGBoost 采用的稀疏感知算法极大的加快了节点分裂的速度；
7. **可以并行化操作：**块结构可以很好的支持并行计算。

**1.3.2 缺点**

1. 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
2. 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

## 2. LightGBM

LightGBM 由微软提出，主要用于解决 GDBT 在海量数据中遇到的问题，以便其可以更好更快地用于工业实践中。

从 LightGBM 名字我们可以看出其是轻量级（Light）的梯度提升机（GBM），其相对 XGBoost 具有训练速度快、内存占用低的特点。下图分别显示了 XGBoost、XGBoost_hist（利用梯度直方图的 XGBoost） 和 LightGBM 三者之间针对不同数据集情况下的内存和训练时间的对比：

![img](https://pic1.zhimg.com/80/v2-e015e3c4018f44787d74a47c9e0cd040_1440w.jpg)

那么 LightGBM 到底如何做到更快的训练速度和更低的内存使用的呢？

我们刚刚分析了 XGBoost 的缺点，LightGBM 为了解决这些问题提出了以下几点解决方案：

1. 单边梯度抽样算法；
2. 直方图算法；
3. 互斥特征捆绑算法；
4. 基于最大深度的 Leaf-wise 的垂直生长算法；
5. 类别特征最优分割；
6. 特征并行和数据并行；
7. 缓存优化。

本节将继续从数学原理和工程实现两个角度介绍 LightGBM。

### 2.1 数学原理

**2.1.1 单边梯度抽样算法**

GBDT 算法的梯度大小可以反应样本的权重，梯度越小说明模型拟合的越好，单边梯度抽样算法（Gradient-based One-Side Sampling, GOSS）利用这一信息对样本进行抽样，减少了大量梯度小的样本，在接下来的计算锅中只需关注梯度高的样本，极大的减少了计算量。

GOSS 算法保留了梯度大的样本，并对梯度小的样本进行随机抽样，为了不改变样本的数据分布，在计算增益时为梯度小的样本引入一个常数进行平衡。具体算法如下所示：

![img](https://pic2.zhimg.com/80/v2-31e5d8d2d0862eda0c40303b3cba6089_1440w.jpg)

我们可以看到 GOSS 事先基于梯度的绝对值对样本进行排序（**无需保存排序后结果**），然后拿到前 a% 的梯度大的样本，和总体样本的 b%，在计算增益时，通过乘上 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1-a%7D%7Bb%7D) 来放大梯度小的样本的权重。**一方面算法将更多的注意力放在训练不足的样本上，另一方面通过乘上权重来防止采样对原始数据分布造成太大的影响。**

**2.1.2 直方图算法**

**1) 直方图算法**

直方图算法的基本思想是将连续的特征离散化为 k 个离散特征，同时构造一个宽度为 k 的直方图用于统计信息（含有 k 个 bin）。利用直方图算法我们无需遍历数据，只需要遍历 k 个 bin 即可找到最佳分裂点。

我们知道特征离散化的具有很多优点，如存储方便、运算更快、鲁棒性强、模型更加稳定等等。对于直方图算法来说最直接的有以下两个优点（以 k=256 为例）：

- **内存占用更小：**XGBoost 需要用 32 位的浮点数去存储特征值，并用 32 位的整形去存储索引，而 LightGBM 只需要用 8 位去存储直方图，相当于减少了 1/8；
- **计算代价更小：**计算特征分裂增益时，XGBoost 需要遍历一次数据找到最佳分裂点，而 LightGBM 只需要遍历一次 k 次，直接将时间复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=+O%28%5C%23data++%2A+%5C%23feature%29+) 降低到 ![[公式]](https://www.zhihu.com/equation?tex=+O%28k++%2A+%5C%23feature%29+) ，而我们知道 ![[公式]](https://www.zhihu.com/equation?tex=%5C%23data+%3E%3E+k) 。

虽然将特征离散化后无法找到精确的分割点，可能会对模型的精度产生一定的影响，但较粗的分割也起到了正则化的效果，一定程度上降低了模型的方差。

**2) 直方图加速**

在构建叶节点的直方图时，我们还可以通过父节点的直方图与相邻叶节点的直方图相减的方式构建，从而减少了一半的计算量。在实际操作过程中，我们还可以先计算直方图小的叶子节点，然后利用直方图作差来获得直方图大的叶子节点。

![img](https://pic2.zhimg.com/80/v2-66982f5386b2e9be3e50a651e01b9c21_1440w.jpg)

**3) 稀疏特征优化**

XGBoost 在进行预排序时只考虑非零值进行加速，而 LightGBM 也采用类似策略：只用非零特征构建直方图。

**2.1.3 互斥特征捆绑算法**

高维特征往往是稀疏的，而且特征间可能是相互排斥的（如两个特征不同时取非零值），如果两个特征并不完全互斥（如只有一部分情况下是不同时取非零值），可以用互斥率表示互斥程度。互斥特征捆绑算法（Exclusive Feature Bundling, EFB）指出如果将一些特征进行融合绑定，则可以降低特征数量。

针对这种想法，我们会遇到两个问题：

1. 哪些特征可以一起绑定？
2. 特征绑定后，特征值如何确定？

**对于问题一：**EFB 算法利用特征和特征间的关系构造一个加权无向图，并将其转换为图着色算法。我们知道图着色是个 NP-Hard 问题，故采用贪婪算法得到近似解，具体步骤如下：

1. 构造一个加权无向图，顶点是特征，边是两个特征间互斥程度；
2. 根据节点的度进行降序排序，度越大，与其他特征的冲突越大；
3. 遍历每个特征，将它分配给现有特征包，或者新建一个特征包，是的总体冲突最小。

算法允许两两特征并不完全互斥来增加特征捆绑的数量，通过设置最大互斥率 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 来平衡算法的精度和效率。EFB 算法的伪代码如下所示：

![img](https://pic3.zhimg.com/80/v2-3eb0ef1f565e344013e8f700fba617da_1440w.jpg)

我们看到时间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28%5C%23feature%5E2%29) ，在特征不多的情况下可以应付，但如果特征维度达到百万级别，计算量则会非常大，为了改善效率，我们提出了一个更快的解决方案：将 EFB 算法中通过构建图，根据节点度来排序的策略改成了根据非零值的技术排序，因为非零值越多，互斥的概率会越大。

**对于问题二：**论文给出特征合并算法，其关键在于原始特征能从合并的特征中分离出来。假设 Bundle 中有两个特征值，A 取值为 [0, 10]、B 取值为 [0, 20]，为了保证特征 A、B 的互斥性，我们可以给特征 B 添加一个偏移量转换为 [10, 30]，Bundle 后的特征其取值为 [0, 30]，这样便实现了特征合并。具体算法如下所示：

![img](https://pic4.zhimg.com/80/v2-ea09eac195f8187917685b8139dc45cf_1440w.jpg)

**2.1.4 带深度限制的 Leaf-wise 算法**

在建树的过程中有两种策略：

- Level-wise：基于层进行生长，直到达到停止条件；
- Leaf-wise：每次分裂增益最大的叶子节点，直到达到停止条件。

XGBoost 采用 Level-wise 的增长策略，方便并行计算每一层的分裂节点，提高了训练速度，但同时也因为节点增益过小增加了很多不必要的分裂，降低了计算量；LightGBM 采用 Leaf-wise 的增长策略减少了计算量，配合最大深度的限制防止过拟合，由于每次都需要计算增益最大的节点，所以无法并行分裂。

![img](https://pic2.zhimg.com/80/v2-76f2f27dd24fc452a9a65003e5cdd305_1440w.jpg)

**2.1.5 类别特征最优分割**

大部分的机器学习算法都不能直接支持类别特征，一般都会对类别特征进行编码，然后再输入到模型中。常见的处理类别特征的方法为 one-hot 编码，但我们知道对于决策树来说并不推荐使用 one-hot 编码：

1. 会产生样本切分不平衡问题，切分增益会非常小。如，国籍切分后，会产生是否中国，是否美国等一系列特征，这一系列特征上只有少量样本为 1，大量样本为 0。这种划分的增益非常小：较小的那个拆分样本集，它占总样本的比例太小。无论增益多大，乘以该比例之后几乎可以忽略；较大的那个拆分样本集，它几乎就是原始的样本集，增益几乎为零；
2. 影响决策树学习：决策树依赖的是数据的统计信息，而独热码编码会把数据切分到零散的小空间上。在这些零散的小空间上统计信息不准确的，学习效果变差。本质是因为独热码编码之后的特征的表达能力较差的，特征的预测能力被人为的拆分成多份，每一份与其他特征竞争最优划分点都失败，最终该特征得到的重要性会比实际值低。

LightGBM 原生支持类别特征，采用 many-vs-many 的切分方式将类别特征分为两个子集，实现类别特征的最优切分。假设有某维特征有 k 个类别，则有 ![[公式]](https://www.zhihu.com/equation?tex=2%5E%7B%28k-1%29%7D+-+1) 中可能，时间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%282%5Ek%29) ，LightGBM 基于 Fisher 大佬的 《[On Grouping For Maximum Homogeneity](https://link.zhihu.com/?target=http%3A//www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf)》实现了 ![[公式]](https://www.zhihu.com/equation?tex=O%28klogk%29) 的时间复杂度。

![img](https://pic3.zhimg.com/80/v2-34558cd9eab486eed731ba7aadca5992_1440w.jpg)

上图为左边为基于 one-hot 编码进行分裂，后图为 LightGBM 基于 many-vs-many 进行分裂，在给定深度情况下，后者能学出更好的模型。

其基本思想在于每次分组时都会根据训练目标对类别特征进行分类，根据其累积值 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Csum+gradient+%7D%7B%5Csum+hessian%7D) 对直方图进行排序，然后在排序的直方图上找到最佳分割。此外，LightGBM 还加了约束条件正则化，防止过拟合。

![img](https://pic2.zhimg.com/80/v2-ea588783be9403a0f7115c408389031d_1440w.jpg)

我们可以看到这种处理类别特征的方式使得 AUC 提高了 1.5 个点，且时间仅仅多了 20%。

### 2.2 工程实现

**2.2.1 特征并行**

传统的特征并行算法在于对数据进行垂直划分，然后使用不同机器找到不同特征的最优分裂点，基于通信整合得到最佳划分点，然后基于通信告知其他机器划分结果。

传统的特征并行方法有个很大的缺点：需要告知每台机器最终划分结果，增加了额外的复杂度（因为对数据进行垂直划分，每台机器所含数据不同，划分结果需要通过通信告知）。

LightGBM 则不进行数据垂直划分，每台机器都有训练集完整数据，在得到最佳划分方案后可在本地执行划分而减少了不必要的通信。

**2.2.2 数据并行**

传统的数据并行策略主要为水平划分数据，然后本地构建直方图并整合成全局直方图，最后在全局直方图中找出最佳划分点。

这种数据划分有一个很大的缺点：通讯开销过大。如果使用点对点通信，一台机器的通讯开销大约为 ![[公式]](https://www.zhihu.com/equation?tex=O%28%5C%23machine+%2A+%5C%23feature+%2A%5C%23bin+%29) ；如果使用集成的通信，则通讯开销为 ![[公式]](https://www.zhihu.com/equation?tex=O%282+%2A+%5C%23feature+%2A%5C%23bin+%29) ，

LightGBM 采用分散规约（Reduce scatter）的方式将直方图整合的任务分摊到不同机器上，从而降低通信代价，并通过直方图做差进一步降低不同机器间的通信。

**2.2.3 投票并行**

针对数据量特别大特征也特别多的情况下，可以采用投票并行。投票并行主要针对数据并行时数据合并的通信代价比较大的瓶颈进行优化，其通过投票的方式只合并部分特征的直方图从而达到降低通信量的目的。

大致步骤为两步：

1. 本地找出 Top K 特征，并基于投票筛选出可能是最优分割点的特征；
2. 合并时只合并每个机器选出来的特征。

**2.2.4 缓存优化**

上边说到 XGBoost 的预排序后的特征是通过索引给出的样本梯度的统计值，因其索引访问的结果并不连续，XGBoost 提出缓存访问优化算法进行改进。

而 LightGBM 所使用直方图算法对 Cache 天生友好：

1. 首先，所有的特征都采用相同的方法获得梯度（区别于不同特征通过不同的索引获得梯度），只需要对梯度进行排序并可实现连续访问，大大提高了缓存命中；
2. 其次，因为不需要存储特征到样本的索引，降低了存储消耗，而且也不存在 Cache Miss的问题。

![img](https://pic4.zhimg.com/80/v2-19436e5546c47fed4a85000b1fff9abb_1440w.jpg)

### 2.3 与 XGBoost 的对比

本节主要总结下 LightGBM 相对于 XGBoost 的优点，从内存和速度两方面进行介绍。

**2.3.1 内存更小**

1. XGBoost 使用预排序后需要记录特征值及其对应样本的统计值的索引，而 LightGBM 使用了直方图算法将特征值转变为 bin 值，且不需要记录特征到样本的索引，将空间复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=O%282%2A%5C%23data%29) 降低为 ![[公式]](https://www.zhihu.com/equation?tex=O%28%5C%23bin%29) ，极大的减少了内存消耗；
2. LightGBM 采用了直方图算法将存储特征值转变为存储 bin 值，降低了内存消耗；
3. LightGBM 在训练过程中采用互斥特征捆绑算法减少了特征数量，降低了内存消耗。

**2.3.2 速度更快**

1. LightGBM 采用了直方图算法将遍历样本转变为遍历直方图，极大的降低了时间复杂度；
2. LightGBM 在训练过程中采用单边梯度算法过滤掉梯度小的样本，减少了大量的计算；
3. LightGBM 采用了基于 Leaf-wise 算法的增长策略构建树，减少了很多不必要的计算量；
4. LightGBM 采用优化后的特征并行、数据并行方法加速计算，当数据量非常大的时候还可以采用投票并行的策略；
5. LightGBM 对缓存也进行了优化，增加了 Cache hit 的命中率。

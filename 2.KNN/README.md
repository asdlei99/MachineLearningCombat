## KNN.py

部分代码说明

### np.tile(A, reps)

复制Ａ矩阵

A：输入矩阵，reps: 参数数组(np.array)

reps: (a, b)

a复制的行数，b重复的次数

### np.meshgrid(x, y)

根据x, y生成网格。

我的理解是:x、y分别是一维数组，y的长度变成x的列长度，构成二维数组。x的长度变为y的横长度，构成二维数组

```python
>>> a = np.array([1, 2])
>>> b = np.array([3, 4, 5])
>>> aa, bb = meshgrid(a, b)
# aa
array([[1, 2],
       [1, 2],
       [1, 2]])
# bb
array([[3, 3],
       [4, 4],
       [5, 5]])

>>> bb, aa = meshgrid(b, a)
# aa
array([[1, 1, 1],
       [2, 2, 2]])
# bb
array([[3, 4, 5],
       [3, 4, 5]])
```

### dict的get方法

dict.get(key, value)

如果key在字典中，则返回对应的value; 若不在则返回value

###　Counter.most_common(n)

返回出现前n个次数最多的

### plt.pcolormesh()

绘制二维数组图，与`pcolor`类似，但是比`pcolor`快

### classify0(inX, dataSet, labels, k)

公式:

$$\sqrt{(A1-A2)^2+(B1-B2)^2+(C1-C2)^2+...}$$

```python
# 1. 距离计算
# inX生成和训练样本对应的矩阵，并与训练样本求差
dataSetSize = dataSet.shape[0]
diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
# 取平方
sqDiffMat = diffMat ** 2
#　求和
sqDistances = sqDiffMat.sum(axis=1)
# 开方
distances = sqDistances ** 0.5
# 从小到大进行排序，返回索引位置
sortedDistIndicies = distances.argsort()
# 2. 选择距离最小的k个点
classCount = {}
for i in range(k):
    # 找到该样本的类型, 并在字典中将该类型加一
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
# 3. 排序并返回出现最多的那个类型
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
return sortedClassCount[0][0]
```

### autoNorm(dataSet)

归一化公式：

$$Y = \frac{X-X_{min}}{X_{max}-X_{min}}$$

其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
```python
normDataSet = np.zeros(np.shape(dataSet))
m = dataSet.shape[0]
# 生成与最小值之差组成的矩阵
normDataSet = dataSet - np.tile(minVals, (m, 1))
# 将最小值之差除以范围组成矩阵
normDataSet = normDataSet / np.tile(ranges, (m, 1))
```

## DecisionTree.py

决策树可视化demo

### 计算香农熵

公式：

$$\sum\limits_{i}=-p_i*\log_2 p_i$$

```python
numEntries = len(dataSet)

# 计算分类标签label出现的次数
labelCounts = {}
for featVec in dataSet:
    # 提取标签值
    currentLabel = featVec[-1]
    # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
    if currentLabel not in labelCounts.keys():
        labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1

# 对于label标签的占比，求出label标签的香农熵
shannonEnt = 0.0
for key in labelCounts:
    # 使用所有类标签的发生频率计算类别出现的概率。
    prob = float(labelCounts[key])/numEntries
    # 计算香农熵，以 2 为底求对数
    shannonEnt -= prob * log(prob, 2)
```

### list.extend()

与`append`不同的是，将括号内元素作为列表加入。等同于列表`＋`

### 切分数据集

切分数据集，如果`index`列值为`value`，则需要剔除此列。

```python
# index -- 表示每一行的index列        划分数据集的特征
retDataSet = []
for featVec in dataSet:
    # 判断index列的值是否为value
    if featVec[index] == value:
        # 剔除index列
        reducedFeatVec = featVec[:index] + featVec[index + 1:]
        # 收集结果值index列为value的行
        retDataSet.append(reducedFeatVec)
```

### 选取最优特征

```python
# 求第一行有多少列的 Feature
numFeatures = len(dataSet[0]) - 1
# label的信息熵
baseEntropy = calcShannonEnt(dataSet)
# 最优的信息增益值, 和最优的Featurn编号
bestInfoGain, bestFeature = 0.0, -1
for i in range(numFeatures):
    # 获取每一个实例的第i+1个feature，组成list集合，去重构成集合
    uniqueVals = set([example[i] for example in dataSet])
    # 创建一个临时的信息熵
    newEntropy = 0.0
    # 遍历某一列的value集合，计算该列的信息熵
    # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))
        newEntropy += prob * calcShannonEnt(subDataSet)
    # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
    # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
    infoGain = baseEntropy - newEntropy
    print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy,
            newEntropy)
    if (infoGain > bestInfoGain):
        bestInfoGain = infoGain
        bestFeature = i
return bestFeature
```

## decisionTreePlot.py

绘制决策数函数
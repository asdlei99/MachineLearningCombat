#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
from math import log
import decisionTreePlot as dtPlot
from collections import Counter
import os

path = os.getcwd()


def createDataSet():
    """
    Desc:
        创建数据集
    Args:
        无需传入参数
    Returns:
        返回数据集和对应的label标签
    """
    # dataSet 前两列是特征，最后一列对应的是每条数据对应的分类标签
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],
               [0, 1, 'no']]
    # labels=[露出水面, 脚蹼]。这里的labels并不参与机器学习
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    r"""
    Desc：
        calculate Shannon entropy -- 计算给定数据集的香农熵
    Args:
        dataSet -- 数据集
    Returns:
        shannonEnt -- 返回 每一组 feature 下的某个分类下，香农熵的信息期望
    公式：
        $\sum\limits_{i}=-p_i*\log_2 p_i$
    """
    # 统计标签出现的次数
    label_count = Counter(data[-1] for data in dataSet)
    # 计算标签的概率
    probs = [p[1] / len(dataSet) for p in label_count.items()]
    # 计算香农熵
    shannonEnt = sum([-p * log(p, 2) for p in probs])
    return shannonEnt


def splitDataSet(dataSet, index, value):
    """
    Desc：
        划分数据集
        splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet  -- 数据集                 待划分的数据集
        index -- 表示每一行的index列        划分数据集的特征
        value -- 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index 列为 value 的数据集【该数据集需要排除index列】
    """

    retDataSet = [
        data[:index] + data[index + 1:] for data in dataSet
        for i, v in enumerate(data) if i == index and v == value
    ]

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    Desc:
        选择切分数据集的最佳特征
    Args:
        dataSet -- 需要切分的数据集
    Returns:
        bestFeature -- 切分数据集的最优的特征列
    """

    # 计算初始香农熵
    base_entropy = calcShannonEnt(dataSet)
    # 最优的信息增益值和最优的Featurn编号
    best_info_gain, best_feature = 0, -1
    # 遍历每一个特征
    for i in range(len(dataSet[0]) - 1):
        # 对当前特征进行统计
        feature_count = Counter([data[i] for data in dataSet])
        # 计算分割后的香农熵
        new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
                       for feature in feature_count.items())
        # 更新值
        info_gain = base_entropy - new_entropy
        # 输出对应特征的更新值
        print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majorityCnt(classList):
    """
    Desc:
        选择出现次数最多的一个结果
    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    major_label = Counter(classList).most_common(1)[0]
    return major_label


def createTree(dataSet, labels):
    """
    Desc:
        创建决策树
    Args:
        dataSet -- 要创建决策树的训练数据集
        labels -- 训练数据集中特征对应的含义的labels，不是目标变量
    Returns:
        myTree -- 创建完成的决策树

    用递归的方法生成决策树, 决策树用字典表示
    """
    # 收集当前所有类别
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    uniqueVals = set([example[bestFeat] for example in dataSet])
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    """
    Desc:
        对新数据进行分类
    Args:
        inputTree  -- 已经训练好的决策树模型
        featLabels -- Feature标签对应的名称，不是目标变量
        testVec    -- 测试输入的数据
    Returns:
        classLabel -- 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对应的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree.get(firstStr)
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    # 保存决策树
    with open(filename, 'w') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    # 加载之前保存的决策树
    with open(filename, 'r') as fr:
        fr = open(filename)
        return pickle.load(fr)


def fishTest():
    """
    Desc:
        对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    Args:
        None
    Returns:
        None
    """
    # 获取数据
    myDat, labels = createDataSet()

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 画图可视化展现
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    """
    Desc:
        预测隐形眼镜的测试代码，并将结果画出来
    Args:
        none
    Returns:
        none
    """

    with open(path + '/Data/lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        # 得到数据的对应的 Labels
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
        lensesTree = createTree(lenses, lensesLabels)
        print(lensesTree)
        # 画图可视化展现
        dtPlot.createPlot(lensesTree)


if __name__ == "__main__":
    # fishTest()
    # ContactLensesTest()
    pass

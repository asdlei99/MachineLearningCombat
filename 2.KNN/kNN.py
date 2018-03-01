#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import operator
import os
from collections import Counter
import numpy as np

path = os.getcwd()

def createDataSet():
    """
    Desc:
        创建数据集和标签
    Args:
        None
    Returns:
        group -- 训练数据集的 features
        labels -- 训练数据集的 labels
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    Desc:
        kNN 的分类函数
    Args:
        inX -- 用于分类的输入向量/测试数据
        dataSet -- 训练数据集的 features
        labels -- 训练数据集的 labels
        k -- 选择最近邻的数目
    Returns:
        sortedClassCount[0][0] -- 输入向量的预测分类 labels
    """
    # 步骤： 计算距离，选取距离最小的k个点，返回出现最多的类型
    dist = np.sum((inX - dataSet)**2, axis=1)**0.5
    k_labels = [labels[index] for index in dist.argsort()[:k]]
    label = Counter(k_labels).most_common(1)[0][0]
    return label

def test():
    group, labels = createDataSet()
    print(group)
    print(labels)
    print(classify0([0.1, 0.1], group, labels, 3))

def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
    """
    with open(filename, 'r') as fr:
        # 获得文件中的数据行的行数
        numberOfLines = len(fr.readlines())
    # 生成对应的零矩阵
    returnMat = np.zeros((numberOfLines, 3)) 
    # print(returnMat)
    classLabelVector = []
    index = 0
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            # 去掉头尾空格，以\t为分隔符切割每一行
            listFromLine = line.strip().split('\t')
            # 每列的属性数据，即 features
            returnMat[index] = listFromLine[:-1]
            # 每列的类别数据，就是 label 标签数据
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    Desc：
        归一化特征值，消除属性之间量级不同导致的影响
    Args：
        dataSet -- 需要进行归一化处理的数据集
    Returns：
        normDataSet -- 归一化处理后得到的数据集
        ranges -- 归一化处理的范围
        minVals -- 最小值
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals) / ranges
    return normDataSet

def datingClassTest():
    """
    Desc：
        对约会网站的测试方法，并将分类错误的数量和分类错误率打印出来
    Args：
        None
    Returns：
        None
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix(path + "/Data/datingTestSet2.txt")
    # 归一化数据
    normMat = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m*hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i], normMat[numTestVecs : m], datingLabels[numTestVecs : m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        # 相当于if, 如果不等于就加1，相等就加0
        errorCount += classifierResult != datingLabels[i]
    print("the total error rate is: %f" % (errorCount / numTestVecs))
    print(errorCount)

def img2vector(filename):
    """
    Desc：
        将图像数据转换为向量
    Args：
        filename -- 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    Returns:
        returnVect -- 图片文件处理完成后的一维矩阵

    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = np.zeros((1, 1024))        
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """
    Desc:
        手写数字识别分类器，并将分类错误数和分类错误率打印出来
    Args:
        None
    Returns:
        None
    """
    # 1. 导入数据
    hwLabels = []
    trainingFileList = os.listdir(path + "/Data/trainingDigits") # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 提取label
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i] = img2vector(path + '/Data/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = os.listdir(path + '/Data/testDigits')  # iterate through the test set
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector(path + '/Data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        errorCount += classifierResult != classNumStr
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / mTest))


if __name__ == '__main__':
    # test()
    # datingClassTest()
    # handwritingClassTest()
    pass
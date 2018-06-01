#!/usr/bin/python
# coding:utf8
'''
Created on Jun 1, 2011
Update  on 2017-12-20
@author: Peter Harrington/片刻
《机器学习实战》更新地址：https://github.com/apachecn/MachineLearning
'''

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    with open(fileName, 'r') as fr:
        # fr = open(fileName)
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [list(map(float, line) for line in stringArr)]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """pca

    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
    """

    # 计算每一列的均值
    meanVals = np.mean(dataMat, axis=0)
    # print ('meanVals', meanVals)

    # 每个向量同时都减去 均值
    meanRemoved = dataMat - meanVals
    # print ('meanRemoved=', meanRemoved)

    covMat = np.cov(meanRemoved, rowvar=0)

    # eigVals为特征值， eigVects为特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # print ('eigVals=', eigVals)
    # print( 'eigVects=', eigVects)
    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # 特征值的逆序就可以得到topNfeat个最大的特征向量
    eigValInd = np.argsort(eigVals)
    # print ('eigValInd1=', eigValInd)

    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    # print ('eigValInd2=', eigValInd)
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    # print ('redEigVects=', redEigVects.T)
    # 将数据转换到新空间
    # print( "---", shape(meanRemoved), shape(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print ('lowDDataMat=', lowDDataMat)
    # print ('reconMat=', reconMat)
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('Data/secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        dataMat[:, 0].flatten().A[0],
        dataMat[:, 1].flatten().A[0],
        marker='^',
        s=90)
    ax.scatter(
        reconMat[:, 0].flatten().A[0],
        reconMat[:, 1].flatten().A[0],
        marker='o',
        s=50,
        c='red')
    plt.show()


def analyse_data(dataMat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    # eigvals, eigVects
    eigvals, _ = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigvals)

    topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score

        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' %
              (format(i + 1, '2.0f'),
               format(line_cov_score / cov_all_score * 100, '4.2f'),
               format(sum_cov_score / cov_all_score * 100, '4.1f')))


if __name__ == "__main__":
    # # 加载数据，并转化数据类型为float
    # dataMat = loadDataSet('Data/testSet.txt')
    # # 只需要1个特征向量
    # lowDmat, reconMat = pca(dataMat, 1)
    # # 只需要2个特征向量，和原始数据一致，没任何变化
    # # lowDmat, reconMat = pca(dataMat, 2)
    # # print (shape(lowDmat))
    # show_picture(dataMat, reconMat)

    # 利用PCA对半导体制造数据降维
    dataMat = replaceNanWithMean()
    print(np.shape(dataMat))
    # 分析数据
    analyse_data(dataMat)
    # lowDmat, reconMat = pca(dataMat, 20)
    # print(shape(lowDmat))
    # show_picture(dataMat, reconMat)

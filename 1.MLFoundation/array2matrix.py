#!/usr/bin/python
# coding:utf-8

import numpy as np

'''
# NumPy 矩阵和数组的区别
NumPy存在2中不同的数据类型:
    1. 矩阵 matrix
    2. 数组 array
相似点：
    都可以处理行列表示的数字元素
不同点：
    1. 2个数据类型上执行相同的数据运算可能得到不同的结果。
    2. NumPy函数库中的 matrix 与 MATLAB中 matrices 等价。
'''

# 生成一个 4*4 的随机数组
randArray = np.random.rand(4, 4)

# 转化关系， 数组转化为矩阵
randMat = np.mat(randArray)

# 从上到下分别为求逆矩阵、转置、矩阵转数组、矩阵转一维数组、共轭矩阵
invRandMat = randMat.I
TraRandMat = randMat.T
ArrRandMat = randMat.A
Ar1RandMat = randMat.A1
ConRandMat = randMat.H
# 输出结果
print('type(randArray) = {} \n{}\n'.format(type(randArray), randArray))
print('type(randMat) = {} \n{}\n'.format(type(randMat), randMat))
print('type(invRandMat) = {} \n{}\n'.format(type(invRandMat), invRandMat))
print('type(TraRandMat) = {} \n{}\n'.format(type(TraRandMat), TraRandMat))
print('type(ArrRandMat) = {} \n{}\n'.format(type(ArrRandMat), ArrRandMat))
print('type(Ar1RandMat) = {} \n{}\n'.format(type(Ar1RandMat), Ar1RandMat))
print('type(ConRandMat) = {} \n{}\n'.format(type(ConRandMat), ConRandMat))
# 矩阵和逆矩阵 进行求积，得到Ｅ(单位矩阵对角线全为１)
myEye = randMat*invRandMat
# 误差
print(myEye - np.eye(4))

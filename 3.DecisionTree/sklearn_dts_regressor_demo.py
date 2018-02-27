#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from numpy.random import RandomState
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 创建一个随机的数据集
# 参考 https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.random.mtrand.RandomState.html
rng = RandomState(1)
# rand() 是给定形状的随机值，rng.rand(80, 1)即矩阵的形状是 80行，1列
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
# print('yyy=', y)

# 拟合回归模型
# 增加 min_samples_leaf=6 的参数，可以提升效果
# 这里的max_depth不设置也可以, 效果基本一致
regr = DecisionTreeRegressor(max_depth=5, min_samples_leaf=6)
regr.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_2 = regr.predict(X_test)

# 绘制结果
plt.figure()
# 绘制散点
plt.scatter(X, y, c="darkorange", label="data")
# 绘制拟合线
plt.plot(
    X_test,
    y_2,
    color="yellowgreen",
    label="max_depth={}".format(regr.max_depth),
    linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

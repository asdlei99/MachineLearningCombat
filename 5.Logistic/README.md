## logistic.py

### 最大似然法

[Logistic回归总结](http://blog.csdn.net/achuo/article/details/51160101)

### 梯度上升的三种方法

1. grad_ascent(data_arr, class_labels):

原始的梯度上升。

目标：找到某个函数的最大值。每次沿函数的梯度方向探寻。一直进行迭代，直到到达某个停止条件（迭代次数限制或某个误差范围）

2. stoc_grad_ascent0(data_mat, class_labels)

随机梯度上升。

不同点在于，第一种方法每次是遍历所有的数据集（一百以内的数据集可以接受用上面的方法）。而随机梯度则是只使用一个样本点来更新回归系数。

3. stoc_grad_ascent1(data_mat, class_labels, num_iter=150)

改进版随机梯度上升。用随机的一个样本来更新回归系数。

### np.ones

```python
# np.ones((10, 1))
array([[ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.]])

# np.ones(10)
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
```

###　np.mat与np.array

都用于转换数组，用法基本一直。mat矩阵可以用于点乘，array数组不能用于点乘

### np.transpose()与np.T

作用都是转置，但是transpose允许添加参数
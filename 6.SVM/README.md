## svm-complete.py

### smoP

```python
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):

        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))

        # 对已存在 alpha对，选出非边界的alpha值，进行优化。
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        iter += 1
```


### np.nonzero()

返回数组（矩阵）中非零元素的位置

```python
>>>x = np.array([[1,0,0], [0,2,0], [1,1,0]])

>>>np.nonzero(x)
# 第一维表示横坐标，第二维表示纵坐标。
# 比如第一个非零元素位置是(0,0),第二个是(1,1)，第三个是(2,0),第四个是(2,1)。
# 取中其中横坐标组成第一维，纵坐标为第二维
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))
```
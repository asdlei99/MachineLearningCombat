## randomForest.py


```python
# 将多个 fold 列表组合成一个 train_set 列表, 类似 union all
In [20]: l1=[[1, 2, 'a'], [11, 22, 'b']]
In [21]: l2=[[3, 4, 'c'], [33, 44, 'd']]
In [22]: l=[]
In [23]: l.append(l1)
In [24]: l.append(l2)
In [25]: l
Out[25]: [[[1, 2, 'a'], [11, 22, 'b']], [[3, 4, 'c'], [33, 44, 'd']]]
In [26]: sum(l, [])
Out[26]: [[1, 2, 'a'], [11, 22, 'b'], [3, 4, 'c'], [33, 44, 'd']]
```
## bayes.py

### 贝叶斯公式

$$p(xy)=p(x|y)p(y)=p(y|x)p(x)$$

$$p(x|y)=\frac{p(y|x)p(x)}{p(y)}$$

### 多维list转一维

```python
# 构造匿名函数
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

# 使用numpy的flatten, 这个的前提是一定要是numpy的array
np.flatten()
```

### 关于二维list转array

会出现失败的情况，原因是list每个维度的长度不定


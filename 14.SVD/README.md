## README

```python
# 计算欧氏距离
myMat = mat(loadExData())
# print(myMat)
print(ecludSim(myMat[:, 0], myMat[:, 4]))
print(ecludSim(myMat[:, 0], myMat[:, 0]))
# 计算余弦相似度
print(cosSim(myMat[:, 0], myMat[:, 4]))
print(cosSim(myMat[:, 0], myMat[:, 0]))
# 计算皮尔逊相关系数
print(pearsSim(myMat[:, 0], myMat[:, 4]))
print(pearsSim(myMat[:, 0], myMat[:, 0]))
```

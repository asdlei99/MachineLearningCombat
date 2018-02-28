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

### 朴素贝叶斯分类原版

```python
def _train_naive_bayes(train_mat, train_category):
    """
    朴素贝叶斯分类原版
    :param train_mat:  type is ndarray
                    总的输入文本，大致是 [[0,1,0,1], [], []]
    :param train_category: 文件对应的类别分类， [0, 1, 0],
                            列表的长度应该等于上面那个输入文本的长度
    p0vec:　非侮辱文件的侮辱单词数与非侮辱文件的侮辱单词总数的对数
    p1vec:　侮辱文件的侮辱单词数与侮辱文件的侮辱单词总数的对数
    pos_abusive:　侮辱文件出现概率
    """
    train_doc_num = len(train_category)
    words_num = len(train_mat[0])
    # 计算侮辱性文件出现概率，对已知分类进行求和（侮辱为1，不是为0），再除以总的句子数（文本数）
    pos_abusive = np.sum(train_category) / train_doc_num
    # 单词出现的次数
    p0num = np.zeros(words_num)
    p1num = np.zeros(words_num)

    # 整个数据集单词出现的次数
    p0num_all = 0
    p1num_all = 0

    for i in range(train_doc_num):
        # 分别统计侮辱性文件和非侮辱性文件的侮辱单词数目，并求和
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    # 后面需要改成改成取 log 函数
    p1vec = p1num / p1num_all
    p0vec = p0num / p0num_all
    return p0vec, p1vec, pos_abusive
```
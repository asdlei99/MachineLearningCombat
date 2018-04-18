#!/usr/bin/python
# coding:utf8

import numpy as np
from numpy.random import shuffle
import matplotlib.pylab as plt
from time import sleep
import bs4
from bs4 import BeautifulSoup as bs
import json
import urllib.request


def loadDataSet(fileName):
    """ 加载数据
        解析以tab键分隔的文件中的浮点数
    Returns：
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的分类标签，即类别标签
    """
    # 获取样本特征的总数，不算最后的目标变量
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            # 删除一行中以tab分隔的数据前后的空白符号
            curLine = line.strip().split('\t')
            # 将数据添加到lineArr List中，每一行数据测试数据组成一个行向量
            lineArr = [float(curLine[i]) for i in range(numFeat - 1)]
            # 将测试数据的输入数据部分(X)存储到dataMat, 类别(label)存储到labelMat
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    '''
    Description：
        线性回归
    Args:
        xArr ：输入的样本数据，包含每个样本数据的 feature
        yArr ：对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        $\beta = (A.T * A)^{-1}*A.T*Y$
        ws：回归系数
    '''

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    # http://www.apache.wiki/pages/viewpage.action?pageId=5505133
    # 书中的公式，求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    Description：
        局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。
    Args：
        testPoint：样本点
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量
        k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关
    Returns:
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点
    Notes:
        这其中会用到计算权重的公式，w = e^((x^((i))-x) / -2k^2)
        理解：x为某个预测点，x^((i))为样本点，样本点距离预测点越近，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）。
        关于预测点的选取，在我的代码中取的是样本点。其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差。
        算法思路：假设预测点取样本点中的第i个样本点（共m个样本点），遍历1到m个样本点（含第i个），算出每一个样本点与预测点的距离，
        也就可以计算出每个样本贡献误差的权值，可以看出w是一个有m个元素的向量（写成对角阵形式）。
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 获得xMat矩阵的行数
    m, _ = np.shape(xMat)
    # 创建权重矩阵weights，该矩阵为每个样本点初始化了一个权重
    weights = np.mat(np.eye((m)))
    for j in range(m):
        # testPoint 的形式是 一个行向量的形式
        # 计算 testPoint 与输入样本点之间的距离，然后下面计算出每个样本贡献误差的权值
        diffMat = testPoint - xMat[j, :]
        # k控制衰减的速度
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    # 根据矩阵乘法计算 xTx ，其中的 weights 矩阵是样本点对应的权重矩阵
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 计算出回归系数的一个估计
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
        Description：
            测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数
        Args：
            testArr：测试所用的所有样本点
            xArr：样本的特征数据，即 feature
            yArr：每个样本对应的类别标签，即目标变量
            k：控制核函数的衰减速率
        Returns：
            yHat：预测点的估计值
    '''
    # 得到样本点的总数
    m, _ = np.shape(testArr)
    # 构建一个全部都是 0 的 1 * m 的矩阵
    yHat = np.zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    # 返回估计值
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):
    '''
    Description:
        首先将 X 排序，其余的都与lwlrTest相同，这样更容易绘图
    Args：
        xArr：样本的特征数据，即 feature
        yArr：每个样本对应的类别标签，即目标变量，实际值
        k：控制核函数的衰减速率的有关参数，这里设定的是常量值 1
    Return：
        yHat：样本点的估计值
        xCopy：xArr的复制
    '''
    # 生成一个与目标变量数目相同的 0 向量
    yHat = np.zeros(np.shape(yArr))
    # 将 xArr 转换为 矩阵形式
    xCopy = np.mat(xArr)
    # 排序
    xCopy.sort(0)
    # 开始循环，为每个样本点进行局部加权线性回归，得到最终的目标变量估计值
    for i in range(np.shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def rssError(yArr, yHatArr):
    '''
        Desc:
            计算分析预测误差的大小
        Args:
            yArr：真实的目标变量
            yHatArr：预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    Desc：
        这个函数实现了给定 lambda 下的岭回归求解。
        如果数据的特征比样本点还多，就不能再使用上面介绍的的线性回归和局部现行回归了，因为计算 (xTx)^(-1)会出现错误。
        如果特征比样本点还多（n > m），也就是说，输入数据的矩阵x不是满秩矩阵。非满秩矩阵在求逆时会出现问题。
        为了解决这个问题，我们下边讲一下：岭回归，这是我们要讲的第一种缩减方法。
    Args：
        xMat：样本的特征数据，即 feature
        yMat：每个样本对应的类别标签，即目标变量，实际值
        lam：引入的一个λ值，使得矩阵非奇异
    Returns：
        $(X^T*X+\alpha*I)^{-1} * X.T*y$
        经过岭回归公式计算得到的回归系数
    '''

    xTx = xMat.T * xMat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    # 检查行列式是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆。
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''
        Desc：
            函数 ridgeTest() 用于在一组 λ 上测试结果
        Args：
            xArr：样本数据的特征，即 feature
            yArr：样本数据的类别标签，即真实数据
        Returns：
            wMat：将所有的回归系数输出到一个矩阵并返回
    '''

    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 计算Y的均值
    # yMean = np.mean(yMat, 0)
    # Y的所有的特征减去均值
    yMat = yMat - np.mean(yMat, axis=0)
    # 标准化 x，计算 xMat 平均值
    # xMeans = np.mean(xMat, 0)
    # 然后计算 X的方差
    # xVar = np.var(xMat, 0)
    # 所有特征都减去各自的均值并除以方差
    xMat = (xMat - np.mean(xMat, axis=0)) / np.var(xMat, axis=0)
    # 可以在 30 个不同的 lambda 下调用 ridgeRegres() 函数。
    numTestPts = 30
    # 创建30 * m 的全部数据为0 的矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    # 按列进行规范化(归一化)
    # 公式： $\frac{X-X.mean()}{X.var()}$=(X-X.mean())/X.var()
    inMat = (xMat.copy() - np.mean(xMat, axis=0)) / np.var(xMat, axis=0)
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean  # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    _, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# def scrapePage(inFile, outFile, yr, numPce, origPrc):
#     fr = open(inFile)
#     fw = open(outFile, 'a')  # a is append mode writing
#     soup = bs(fr.read())
#     i = 1
#     currentRow = soup.findAll('table', r="%d" % i)
#     while (len(currentRow) != 0):
#         title = currentRow[0].findAll('a')[1].text
#         lwrTitle = title.lower()
#         if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
#             newFlag = 1.0
#         else:
#             newFlag = 0.0
#         soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
#         if len(soldUnicde) == 0:
#             print("item #%d did not sell" % i)
#         else:
#             soldPrice = currentRow[0].findAll('td')[4]
#             priceStr = soldPrice.text
#             priceStr = priceStr.replace('$', '')  # strips out $
#             priceStr = priceStr.replace(',', '')  # strips out ,
#             if len(soldPrice) > 1:
#                 priceStr = priceStr.replace('Free shipping', '')  # strips out Free Shipping
#             print("%s\t%d\t%s" % (priceStr, newFlag, title))
#             fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr, numPce, newFlag, origPrc, priceStr))
#         i += 1
#         currentRow = soup.findAll('table', r="%d" % i)
#     fw.close()

# --------------------------------------------------------------
# 预测乐高玩具套装的价格 ------ 最初的版本，因为现在 google 的 api 变化，无法获取数据
# 故改为了下边的样子，但是需要安装一个 beautifulSoup 这个第三方网页文本解析器，安装很简单，见下边
# from time import sleep
# import json
# 这里特别指出 正确的使用方法为下面的语句使用,from urllib import request 将会报错,具体细节查看官方文档
# import urllib.request   # 在Python3中将urllib2和urllib等五个模块合并为一个标准库urllib,其中的urllib2.urlopen更改为urllib.request.urlopen


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())  # 转换为json格式
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            newFlag = 1 if currItem['product']['condition'] == 'new' else 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc,
                                                  sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


# 交叉验证
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    # 创造错误矩阵
    #create error mat 30columns numVal rows创建error mat 30columns numVal 行
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        shuffle(indexList)
        for j in range(m):
            #基于indexList中的前90%的值创建训练集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  #get 30 weight vectors from ridge
        for k in range(30):  #loop over all of the ridge estimates
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)

            #regularize test with training params
            # 测试集与训练集的平均值，除以训练集方差
            matTestX = (
                matTestX - np.mean(matTrainX, 0)) / np.var(matTrainX, 0)
            #test ridge results and store
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))
            #print (errorMat[i,k])
            #calc avg performance of the different ridge weight vectors
    meanErrors = np.mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ",
          -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))


# 预测乐高玩具套装的价格 可运行版本，我们把乐高数据存储到了我们的 input 文件夹下，使用 urllib爬取,bs4解析内容


# 从页面读取数据，生成retX和retY列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    with open(inFile) as fr:
        soup = bs(fr.read())
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')  #strips out $
            priceStr = priceStr.replace(',', '')  #strips out ,
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc,
                                              sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)


'''
# 依次读取六种乐高套装的数据，并生成数据矩阵        
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'Data/setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'Data/setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'Data/setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'Data/setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'Data/setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'Data/setHtml/lego10196.html', 2009, 3263, 249.99)
# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):
    # 获得数据点个数，xArr和yArr具有相同长度
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))
    # 主循环 交叉验证循环
    for i in range(numVal):
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        trainX=[]; trainY=[]
        testX = []; testY = []
        # 对数据进行混洗操作
        random.shuffle(indexList)
        # 切分训练集和测试集
        for j in range(m):
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        # 获得回归系数矩阵
        wMat = ridgeTest(trainX,trainY)
        # 循环遍历矩阵中的30组回归系数
        for k in range(30):
            # 读取训练集和数据集
            matTestX = mat(testX); matTrainX=mat(trainX)
            # 对数据进行标准化
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            # 测试回归效果并存储
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)
            # 计算误差
            errorMat[i,k] = ((yEst.T.A-array(testY))**2).sum()
    # 计算误差估计值的均值
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    # 输出构建的模型
    print ("the best model from Ridge Regression is:\n",unReg)
    print ("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))

'''


# test for standRegression
def regression1():
    xArr, yArr = loadDataSet("Data/data.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(
        111)  # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
    ax.scatter(
        [xMat[:, 1].flatten()],
        [yMat.T[:, 0].flatten().A[0]])  # scatter 的x是xMat中的第二列，y是yMat的第一列
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


def regression2():
    xArr, yArr = loadDataSet("Data/data.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    srtInd = xMat[:, 1].argsort(
        0)  # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(
        [xMat[:, 1].flatten().A[0]], [np.mat(yArr).T.flatten().A[0]],
        s=2,
        c='red')
    plt.show()


# test for abloneDataSet
def abaloneTest():
    '''
    Desc:
        预测鲍鱼的年龄
    Args:
        None
    Returns:
        None
    '''
    # 加载数据
    abX, abY = loadDataSet("Data/abalone.txt")
    # 使用不同的核进行预测
    oldyHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    oldyHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    oldyHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # 打印出不同的核预测值与训练数据集上的真实值之间的误差大小
    print("old yHat01 error Size is :", rssError(abY[0:99], oldyHat01.T))
    print("old yHat1 error Size is :", rssError(abY[0:99], oldyHat1.T))
    print("old yHat10 error Size is :", rssError(abY[0:99], oldyHat10.T))

    # 打印出 不同的核预测值 与 新数据集（测试数据集）上的真实值之间的误差大小
    newyHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("new yHat01 error Size is :", rssError(abY[0:99], newyHat01.T))
    newyHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("new yHat1 error Size is :", rssError(abY[0:99], newyHat1.T))
    newyHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("new yHat10 error Size is :", rssError(abY[0:99], newyHat10.T))

    # 使用简单的 线性回归 进行预测，与上面的计算进行比较
    standWs = standRegres(abX[0:99], abY[0:99])
    standyHat = np.mat(abX[100:199]) * standWs
    print("standRegress error Size is:", rssError(abY[100:199], standyHat.T.A))


# test for ridgeRegression
def regression3():
    abX, abY = loadDataSet("Data/abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


# test for stageWise
def regression4():
    xArr, yArr = loadDataSet("Data/abalone.txt")
    stageWise(xArr, yArr, 0.01, 200)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = regularize(xMat)
    yM = np.mean(yMat, 0)
    yMat = yMat - yM
    weights = standRegres(xMat, yMat.T)
    print(weights.T)


# predict for lego's price
def regression5():
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY, 10)


if __name__ == '__main__':
    # regression1()
    # regression2()
    # abaloneTest()
    # regression3()
    # regression4()
    # regression5()
    pass
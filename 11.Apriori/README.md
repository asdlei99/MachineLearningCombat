
## 函数说明

- frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用

- s.issubset(t)  测试是否 s 中的每一个元素都在 t 中， s是集合， t可以是列表、集合等

```python

"""
def getActionIds():

    # votesmart.apikey = 'get your api key first'
    votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
    actionIdList = []
    billTitleList = []
    fr = open('Data/recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) # api call
            for action in billDetail.actions:
                if action.level == 'House' and (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print ('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print ("problem getting bill %d" % billNum)
        sleep(1)                                      # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if vote.candidateName not in transDict.keys():
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
"""

# 暂时没用上
# def pntRules(ruleList, itemMeaning):
#     for ruleTup in ruleList:
#         for item in ruleTup[0]:
#             print itemMeaning[item]
#         print "           -------->"
#         for item in ruleTup[1]:
#             print itemMeaning[item]
#         print "confidence: %f" % ruleTup[2]
#         print       #print a blank line


```
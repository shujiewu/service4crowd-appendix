
import numpy as np
import sys
import json

from scipy import stats
from PolicyLiveNewExpertChangeOb import QualityPOMDPPolicy
import math
import random
from math import sqrt

def normalize(array):
    sum = 0.0
    for i in range(0, len(array)):
        sum += array[i]
    for i in range(0, len(array)):
        array[i] = array[i] / sum
    return array
def findBestAction(belief):
    bestValue = -1230981239102938019
    bestAction = 0  # Assume there is at least one action
    for action in range(0,2):
        value = findBestValue(belief, q.policy[action])
        if value > bestValue:
            bestValue = value
            bestAction = action
    return bestAction
def findBestValue(belief, hyperplanes):
    bestValue = -129837198273981231
    for hyperplane in hyperplanes:
        dontUse = False
        for (b, entry) in zip(belief, hyperplane):
            if entry == '*':
                dontUse = True
                # print(1111)
                break
        if dontUse:
            continue
        # print(hyperplane)
        value = np.dot(belief, hyperplane)
        if value > bestValue:
            bestValue = value
    return bestValue

def calcAccuracy(gamma, d):
    return (1.0 / 2) * (1.0 + (1.0 - d) ** gamma)

def getMax(state):
    if state[0] > state[1]:
        return state[0]
    else:
        return state[1]

def getMin(state):
    if state[0] <= state[1]:
        return state[0]
    else:
        return state[1]

def computeTransWithAction(state,workerPara,workerId,d,workerAbility):
    transDict ={}
    for (index, i) in enumerate(state):
        q = i[1]
        value = []
        trans = []
        for (index2, j) in enumerate(state):
            if j[0] == q:
                a = workerPara[workerId]['coef1']
                b = workerPara[workerId]['coef2']
                intercept = workerPara[workerId]['intercept']
                e = workerPara[workerId]['mse']
                mean = a * q + b * d +intercept
                pr = stats.norm.pdf(j[1], loc=mean, scale=e)  # stats.beta.pdf(x=q2, a=2, b=1)
                trans.append(index2)
                if j[0] != j[1]:
                    value.append(pr)
                else:
                    NoChange = ((1.0 / 2) * (1.0 + (j[0]) ** (workerAbility[workerId] / j[0])))
                    value.append(NoChange + (1 - NoChange) * pr)
        value = normalize(value)
        target = {}
        for (index3, s) in enumerate(value):
            target[str(trans[index3])] = s
            # file.write('T: %d : %d : %d %f\n' % (0, index, trans[index3], s))
        ## index state可以转移到其他的target位置的概率
        transDict[str(index)] = target
    return transDict

def updateBeliefWithAction(belief,workerAbility,workerId,trans,change):
    newbelief = []
    for (index,v) in enumerate(range(1, 21, 1)):
        q1 = v / 20
        # pr1 = stats.beta.pdf(x=q1, a=5, b=1)
        for (index2,v2) in enumerate(range(1, 21, 1)):
            q2 = v2 / 20
            # pr2 = stats.beta.pdf(x=q2, a=101, b=1)
            if q2!=q1:
                po = 1.0
                # print(po)
            else:
                NoChange = ((1.0 / 2) * (1.0 + (q1) ** (workerAbility[workerId] / q1)))
                if change==False:
                    po =  NoChange
                else:
                    po = 1-NoChange
                # po = 1-q1
            targetIndex = index*20+index2
            sumProb = 0.0
            for i in trans:
                for ii in trans[i]:
                    if int(ii)==targetIndex:
                        sumProb = sumProb + trans[i][ii]*belief[int(i)]
            newbelief.append(po*sumProb)
    newbelief.append(0)
    belief = normalize(newbelief)
    return belief

def readParameter(file):
    with open(file, 'r') as load_f:
        worker = json.load(load_f)
    return worker
def readWorkerAbility(file):
    with open(file, 'r') as load_f:
        worker_a = json.load(load_f)
    return worker_a
def readTask(file):
    with open(file, 'r') as load_f:
        task = json.load(load_f)
    return task

if __name__ == "__main__":
    estimateResult = str(sys.argv[1])
    workerAbilityFile = str(sys.argv[2])
    transParameterFile= str(sys.argv[3])
    cost = str(sys.argv[4])
    storage_dir = str(sys.argv[5])

    q = QualityPOMDPPolicy()
    ###16的结果很好
    ###20 0.2 21 0.1 23 0.3
    # 20 多样化
    q.readPolicy('./out0.25.policy')
    ### 26:0.1 27:0.2 28:0.3
    workerPara = readParameter(transParameterFile)
    workerAbility = readWorkerAbility(workerAbilityFile)
    tasks = readTask(estimateResult)
    workerTrans = {}

    state = []
    for v in range(1, 21, 1):
        q1 = v / 20
        for v2 in range(1, 21, 1):
            q2 = v2 / 20
            vv = []
            vv.append(q1)
            vv.append(q2)
            state.append(vv)
    workerList = []
    for worker in workerAbility:
        if int(worker)<100:
            workerList.append(worker)

    it = 0
    sum = 0
    sum2 = 0
    ballot = []
    sumballot1 = 0
    sumballot2 = 0
    quality = 0.0
    moreQuality = 0.0

    decision=[]
    submitNum = 0
    createNum = 0
    noIterList=[]
    moreIterList=[]

    for (idx,taskJson) in enumerate(tasks):
        task_id = '1'
        for key in taskJson:
            task = taskJson[key]
            task_id = key
            # task = taskJson
        init = task[0]
        second = task[1]

        ### init
        belief = []
        for v in range(1, 21, 1):
            q1 = v / 20
            pr1 = stats.norm.pdf(q1,loc=1-init['difficult'], scale=0.01)
            for v2 in range(1, 21, 1):
                q2 = v2 / 20
                vv = []
                vv.append(q1)
                vv.append(q2)
                a = workerPara[second['workerId']]['coef1']
                b = workerPara[second['workerId']]['coef2']
                e = workerPara[second['workerId']]['mse']
                intercept = workerPara[second['workerId']]['intercept']
                mean = a * q1 + b*init['difficult'] +intercept
                # mean = a*workerAbility[second['workerId']]+b
                pr2 = stats.norm.pdf(q2,loc=mean, scale=e) #stats.beta.pdf(x=q2, a=2, b=1)
                belief.append(pr1 * pr2)
        belief.append(0)
        belief = normalize(belief)
        estimatedBallotsToCompletion = []

        iter = 2
        for simulation in range(1):
            estimatedBallotsToCompletion.append(0)
            # Take ballots until the POMDP submits
            difficult = 0.05
            first = False
            while True:
                bestAction = findBestAction(belief)
                if bestAction > 0:
                    if bestAction ==2:
                        print("expert")
                    if bestAction ==1:
                        print("complete")
                        estimatedBallotsToCompletion[-1] += iter - 1
                    break
                change = False
                if iter<len(task):
                    ## 看第二个人
                    workerId = task[iter]['workerId']
                    change = task[iter]['change']
                    difficult = task[iter-1]['difficult']
                else:
                    iter = iter + 1
                    break
                iter =iter+1
                trans = computeTransWithAction(state, workerPara, workerId,difficult,workerAbility)
                belief = updateBeliefWithAction(belief,workerAbility,workerId,trans,change)
        if iter <= len(task):
            submitNum = submitNum+1
            noIterList.append(task_id)
            # if model['workerType'] == 'model1':
            #     model1stat[model['workerId']]['total'] += estimatedBallotsToCompletion[-1]
            #     model1stat[model['workerId']]['num'] +=1
            #     print(model1stat)
            # elif model['workerType'] == 'model2':
            #     model2stat[model['workerId']]['total'] += estimatedBallotsToCompletion[-1]
            #     model2stat[model['workerId']]['num'] +=1
            #     print(model2stat)
        else:
            createNum = createNum + 1
            moreIterList.append(task_id)
        # ballot.append(estimatedBallotsToCompletion[-1])
        # if it ==100:
        #     break
        # for index,answer in enumerate(task):

    # print(ballot)
    # print(sum/it)
    # print(sumballot1)
    # print(quality/sumballot1)
    # print()
    # print(sum2 / sumballot2)
    # print(sumballot2)
    # print(moreQuality/sumballot2)
    # print(model1stat)
    # print(model2stat)

    file_name = storage_dir+'/output'
    res = {}
    res['noIterNum']=submitNum
    res['moreIterNum'] = createNum
    res['noIterList'] = noIterList
    res['moreIterList'] = moreIterList
    with open(file_name,'w') as file_obj:
        json.dump(res,file_obj)
    print(res)






import sys

sys.path.insert(0, '../')
import os
import subprocess
from itertools import product
from scipy import stats
import math
import numpy as np
from math import sqrt
class QualityPOMDPPolicy(object):
    policyType = 0
    path = 'E:/service4crowd/FastWash/detection_tools/fastwash_model'
    numStates = 401
    numDiffs = 11
    numActions = 3

    debug = False
    fastLearning = False
    timeLearning = 900

    if sys.platform == 'darwin':
        ZMDPPath = 'E:/service4crowd/FastWash/detection_tools/fastwash_model/zmdp-1.1.7/bin/darwin/zmdp'
    else:
        ZMDPPath = 'E:/service4crowd/FastWash/detection_tools/fastwash_model/zmdp-1.1.7/bin/darwin/zmdp'
    actions = range(0, 3)

    # difficulties = getDifficulties(0.1)
    def normalize(self, array):
        sum = 0.0
        for i in range(0, len(array)):
            sum += array[i]
        for i in range(0, len(array)):
            array[i] = array[i] / sum
        return array

    def __init__(self, value=100, price=1):

        self.averageGamma = 0.5
        self.value = value
        self.price = price
        self.policy = None
        # self.learnPOMDP()

    def learnPOMDP(self):
        os.chdir(QualityPOMDPPolicy.path)
        if os.path.exists(
                'log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value, self.price, self.averageGamma)):
            self.readPolicy(
                'log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value, self.price, self.averageGamma))
            return

        print("Learning Quality POMDP policy.")
        self.genPOMDP('log/pomdp/%d_%4.2f_%4.2f.pomdp' % (self.value, self.price, self.averageGamma))
        # Solve the POMDP
        zmdpDumpfile = open(
            'E:/service4crowd/FastWash/detection_tools/fastwash_model/log/pomdp/%d_%4.2f_%4.2f.zmdpDump' % (self.value, self.price, self.averageGamma), 'w')
        command = '%s solve %s -o %s -t %d' % (
            QualityPOMDPPolicy.ZMDPPath,
            QualityPOMDPPolicy.path + '/log/pomdp/%d_%4.2f_%4.2f.pomdp' % (
            self.value, self.price, self.averageGamma),
            QualityPOMDPPolicy.path + '/log/pomdp/%d_%4.2f_%4.2f.policy' % (
            self.value, self.price, self.averageGamma),
            QualityPOMDPPolicy.timeLearning)
        subprocess.call(command,
                        stdout=zmdpDumpfile)
                        # shell=True,
                        # executable="/bin/bash")
        zmdpDumpfile.close()
        print("Learned!")
        # Read the policy that we will begin with
        os.chdir(QualityPOMDPPolicy.path)
        self.readPolicy(
            'log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value, self.price, self.averageGamma))
        ################################################################
        # If you want a policy that we've already learned
        # pick one from ModelLearning/
        # W2R100 means 2 workflows, value of answer is 100
        #################################################################
        # policy = readPolicy("ModelLearning/Policies/W2R100.policy",
        #                    numStates)
        # print "POMDP Generated, POMDP Solved, Policy read"

    ############################
    # There are (numDiffs) * 2 states +  1 terminal state at the end.
    # We index as follows: Suppose Type A is the 0th difficulty,
    # type B is the 5th difficulty, and the answer is zero.
    # Then, the corresponding state number is
    # (0 * numDiffs * numDiffs) + (0 * numDiffs) + 5.
    #
    # Essentially, the first half of the states represents answer zero
    # The second half represents answer one
    # Each half is divided into numDiffs sections, representing
    # each possible difficulty for a typeA question.
    # Then each section is divided into numDiffs sections, representing
    # each possible difficulty for a typeB question.
    ###########################
    def calcAccuracy(self, gamma, d):
        return (1.0 / 2) * (1.0 + (1.0 - d) ** gamma)

    def getMax(self, state):
        if state[0] > state[1]:
            return state[0]
        else:
            return state[1]
    def getMin(self, state):
        if state[0] <= state[1]:
            return state[0]
        else:
            return state[1]
    def genPOMDP(self, filename):

        # Add one absorbing state
        file = open(filename, 'w')
        file.write('discount: 0.99\n')
        file.write('values: reward\n')
        file.write('states: %d\n' % QualityPOMDPPolicy.numStates)
        file.write('actions: %d\n' % QualityPOMDPPolicy.numActions)

        SUBMIT = 1
        SUBMITExpert = 2
        file.write('observations: Change NoChange None\n')

        # Taking Action "Ask for ballot" always keeps you in the same state
        # for i in range(0, QualityPOMDPPolicy.numStates):
        #     file.write('T: %d : %d : %d %f\n' % (0, i, i, 1.0))

        state = []
        for v in range(1, 21, 1):
            q1 = v / 20
            for v2 in range(1, 21, 1):
                q2 = v2 / 20
                vv = []
                vv.append(q1)
                vv.append(q2)
                state.append(vv)

        for (index, i) in enumerate(state):
             q = i[1]
             prselectQ = 1.0
             value = []
             trans = []
             for (index2, j) in enumerate(state):
                 if j[0] == q:
                     a = 0.3176203240922768
                     b = -0.10024098457030137
                     e = 0.08466120444843533
                     d = 0.1
                     intercept = 0.6601884131570299
                     mean = a * q + b * d + intercept
                     pr = stats.norm.pdf(j[1], loc=mean, scale=e)  # stats.beta.pdf(x=q2, a=2, b=1)
                     trans.append(index2)
                     if j[0]!=j[1]:
                         value.append(pr)
                     else:
                         NoChange = ((1.0 / 2) * (1.0 + (j[0]) ** (0.7 / j[0])))
                         value.append(j[0]+(1-j[0])*pr)
             value = self.normalize(value)
             for (index3, s) in enumerate(value):
                 file.write('T: %d : %d : %d %f\n' % (0, index, trans[index3], s))


        # Add transitions to absorbing state
        file.write('T: %d : %d : %d %f\n' % (0, QualityPOMDPPolicy.numStates - 1, QualityPOMDPPolicy.numStates - 1, 1.0))
        file.write('T: %d : %d : %d %f\n' % (2, QualityPOMDPPolicy.numStates - 1, QualityPOMDPPolicy.numStates - 1, 1.0))
        file.write('T: %d : * : %d %f\n' % (SUBMIT, QualityPOMDPPolicy.numStates - 1, 1.0))


        # file.write('T: %d : * : %d %f\n' % (SUBMITExpert, QualityPOMDPPolicy.numStates - 1, 1.0))
        for (index, i) in enumerate(state):
             q = i[1]
             prselectQ = 1.0
             value = []
             trans = []
             for (index2, j) in enumerate(state):
                 if j[0] == q:
                     if j[1] ==1.0:
                        trans.append(index2)
                        value.append(1.0)
             value = self.normalize(value)
             for (index3, s) in enumerate(value):
                 file.write('T: %d : %d : %d %f\n' % (SUBMITExpert, index, trans[index3], s))

        # Add observations in absorbing state
        file.write('O: * : %d : None %f\n' % (QualityPOMDPPolicy.numStates - 1, 1.0))

        for (index, i) in enumerate(state):
            file.write('O: %d: %d : None %f\n' % (SUBMIT, index, 1.0))
            file.write('O: %d: %d : None %f\n' % (SUBMITExpert, index, 1.0))
            if i[0] != i[1]:
                file.write('O: %d: %d : Change %f\n' % (0, index, 1.0))
                file.write('O: %d: %d : NoChange %f\n' % (0, index, 0.0))
            else:
                NoChange = ((1.0 / 2) * (1.0 + (i[0]) ** (0.7/i[0])))
                file.write('O: %d: %d : Change %f\n' % (0, index, 1-NoChange))
                file.write('O: %d: %d : NoChange %f\n' % (0, index, NoChange))
                # file.write('O: %d: %d : Last %f\n' % (
                # 0, index, self.calcAccuracy(i[1]/0.7, 1 - math.pow(math.fabs(i[0] - i[1]), 0.5))))
                # file.write('O: %d: %d : Current %f\n' % (
                #     0, index, 1.0-self.calcAccuracy(i[1]/0.7, 1 - math.pow(math.fabs(i[0] - i[1]), 0.5))))
        # for v in range(0, 2):
        #     for diffState in product(range(QualityPOMDPPolicy.numDiffs), repeat = 1):
        #         state = v * QualityPOMDPPolicy.numDiffs
        #         for k in range(0, 1):
        #             state += (diffState[k] * (QualityPOMDPPolicy.numDiffs ** (1 - (k+1))))
        #         file.write('O: %d: %d : None %f\n' % (SUBMITZERO, state, 1.0))
        #         file.write('O: %d: %d : None %f\n' % (SUBMITONE, state, 1.0))
        #         if v == 0: #if the answer is 0
        #             for k in range(0, 1):
        #                 file.write('O: %d : %d : Zero %f\n' %(k, state,calcAccuracy(gammas[k], QualityPOMDPPolicy.difficulties[diffState[k]])))
        #                 file.write('O: %d : %d : One %f\n' %(k, state, 1.0 - calcAccuracy(gammas[k],QualityPOMDPPolicy.difficulties[diffState[k]])))
        #         else: # if the answer is 1
        #             for k in range(0, 1):
        #                 file.write('O: %d : %d : Zero %f\n' %(k, state,1.0 - calcAccuracy(gammas[k], QualityPOMDPPolicy.difficulties[diffState[k]])))
        #                 file.write('O: %d : %d : One %f\n' %(k, state, calcAccuracy(gammas[k],QualityPOMDPPolicy.difficulties[diffState[k]])))

        ## file.write('R: * : * : * : * %f\n' % (-1 * self.price))
        # file.write('R: * : * : * : * %f\n' % (-1 * self.price))
        print(state)
        for (index1, i) in enumerate(state):
            ## 转换为另一种
            q = i[1]
            trans = []
            deltaq = []
            max = []
            print(i)
            for (index2, j) in enumerate(state):
                if j[0] == q:
                    trans.append(index2)
                    deltaq.append(j[1]-j[0])
                    max.append(j[1])
                    # if j[0]<j[1]:
                    #
                    # else:
                    #     max.append(j[0])
            print(deltaq)
            self.price = 2.0
            for (index,next) in enumerate(trans):
                # print((math.exp(deltaq[index]) - 1) )/ (math.exp(1) - 1))
                # print((math.exp(deltaq[index])-1)/(math.exp(1)-1))
                # file.write('R: %d : %d : %d : * %f\n' % (0, index, next, -1*self.price+(math.exp(deltaq[index])-1)/(math.exp(1)-1)))
                if deltaq[index]>=0:
                    file.write('R: %d : %d : %d : * %f\n' % (0, index1, next, -1 * self.price + 10*(math.exp(deltaq[index])-1)/(math.exp(1)-1)))
                else:
                    file.write('R: %d : %d : %d : * %f\n' % (0, index1, next, -1 * self.price - (math.exp(math.fabs(deltaq[index])-1))/(math.exp(1)-1)))
            file.write('R: %d : %d : %d : * %f\n' % (SUBMIT, index1, QualityPOMDPPolicy.numStates - 1, 10*(math.exp(q)-1)/(math.exp(1)-1)))

        for (index1, i) in enumerate(state):
             q = i[1]
             deltaq = []
             trans = []
             for (index2, j) in enumerate(state):
                 if j[0] == q:
                     if j[1] ==1.0:
                        trans.append(index2)
                        deltaq.append(j[1] - j[0])
             for (index, next) in enumerate(trans):
                 file.write('R: %d : %d : %d : * %f\n' % (2, index1, next, -150000000 * self.price + 5*(math.exp(deltaq[index]) - 1) / (math.exp(1) - 1)))
        # Add rewards in absorbing state
        file.write('R: * : %d : %d : * %f\n' % (QualityPOMDPPolicy.numStates - 1, QualityPOMDPPolicy.numStates - 1, 0))

        # for i in range(0, QualityPOMDPPolicy.numStates - 1):
        #     if i < (QualityPOMDPPolicy.numStates - 1) / 2:
        #         file.write('R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, QualityPOMDPPolicy.numStates - 1, 1))
        #         file.write(
        #             'R: %d : %d : %d : * %f\n' % (SUBMITONE, i, QualityPOMDPPolicy.numStates - 1, -1 * self.value))
        #     else:
        #         file.write('R: %d : %d : %d : * %f\n' % (SUBMITONE, i, QualityPOMDPPolicy.numStates - 1, 1))
        #         file.write(
        #             'R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, QualityPOMDPPolicy.numStates - 1, -1 * self.value))

        file.close()

    def readPolicy(self, pathToPolicy):
        policy = {}
        lines = open(pathToPolicy, 'r').read().split("\n")

        numPlanes = 0
        action = 0
        alpha = [0 for k in range(0, QualityPOMDPPolicy.numStates)]
        insideEntries = False
        for i in range(0, len(lines)):
            line = lines[i]
            # First we ignore a bunch of lines at the beginning
            if (line.find('#') != -1 or line.find('{') != -1 or
                    line.find('policyType') != -1 or line.find('}') != -1 or
                    line.find('numPlanes') != -1 or
                    ((line.find(']') != -1) and not insideEntries) or
                    line.find('planes') != -1 or line == ''):
                continue
            if line.find('action') != -1:
                words = line.strip(', ').split(" => ")
                action = int(words[1])
                continue
            if line.find('numEntries') != -1:
                continue
            if line.find('entries') != -1:
                insideEntries = True
                continue
            if (line.find(']') != -1) and insideEntries:  # We are done with one alpha vector
                if action not in policy:
                    policy[action] = []
                policy[action].append(alpha)
                action = 0
                alpha = ['*' for k in range(0, QualityPOMDPPolicy.numStates)]
                numPlanes += 1
                insideEntries = False
                continue
            # If we get here, we are reading state value pairs
            entry = line.split(",")
            state = int(entry[0])
            val = float(entry[1])
            alpha[state] = val
        # print "Policy Read"
        self.policy = policy
if __name__ == '__main__':
    q= QualityPOMDPPolicy()
    q.learnPOMDP()
# print(stats.beta.cdf(x=1, a=2, b=1))
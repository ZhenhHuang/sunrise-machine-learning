'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # x0-x1 y0-y1

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 处理成一行
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda k: k[1], reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去首尾空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {}, the real answer is:{} ".format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: ", (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir(
        r'C:\Users\Superclass\Documents\quantitative_investment\code\codes\machinelearninginaction\Ch02\EXTRAS\trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector(
            r'C:\Users\Superclass\Documents\quantitative_investment\code\codes\machinelearninginaction\Ch02\EXTRAS\trainingDigits\%s' % fileNameStr)
    testFileList = listdir(
        r'C:\Users\Superclass\Documents\quantitative_investment\code\codes\machinelearninginaction\Ch02\EXTRAS\testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(
            r'C:\Users\Superclass\Documents\quantitative_investment\code\codes\machinelearninginaction\Ch02\EXTRAS\testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: {}, the real answer is:{} ".format(classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: {}".format(errorCount))
    print("\nthe total error rate is: ", (errorCount / float(mTest)))


# a,b = createDataSet()
# print(classify0([0,0.2],a,b,3))
# 
# handwritingClassTest()

def train(fileName, k):
    data, labels = file2matrix(fileName)
    data,r,m = autoNorm(data)

    length = data.shape[0]
    dataTrain = data[0:int(length * 0.8)]
    dataTest = data[int(length * 0.8):]
    labelsTrain = labels[0:int(length * 0.8)]
    labelsTest = labels[int(length * 0.8):]
    correct = 0.0
    labelsForOutput = []
    for d in dataTest:
        labelsForOutput.append(classify0(d, dataTrain, labelsTrain, k))
    for i in range(len(labelsForOutput)):
        if labelsForOutput[i] == labelsTest[i]:
            correct += 1
    return correct / len(labelsForOutput)


if __name__ == '__main__':
    accuracy = []
    for i in range(10):
        if i == 0:
            pass
        else:
            accuracy.append(train(r"datingTestSet.txt", i))
    import matplotlib.pyplot as plt

    plt.ylabel("accuracy")
    plt.xlabel("K")
    plt.plot([i for i in range(10) if i != 0], accuracy)
    plt.show()

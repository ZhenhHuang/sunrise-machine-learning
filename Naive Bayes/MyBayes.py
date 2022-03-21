import random

import numpy as np
import re
import os


class MyBayes:
    def __init__(self, path: list):
        self.path = path
        self.Tokens = None
        self.vocab = None
        self.testSet = None
        self.testLabel = None
        self.p1 = 0
        self.p2 = 0
        self.p1_prior = 0

    @staticmethod
    def loadData(path, flag='ham'):
        Tokens = []
        classList = []
        file_list = os.listdir(path)
        c = 1
        if path.split('/')[1] == flag:
            c = 1
        else:
            c = 0
        for file in file_list:
            text = open(path + '/' + file).read()
            TokenList = re.split(r'\W+', text)
            TokenList = [tok.lower() for tok in TokenList if len(tok) > 2]
            classList.append(c)
            Tokens.append(TokenList)
        return Tokens, classList

    def split_train_test(self, ratio):
        ham_tokens, ham_list = self.loadData(self.path[0])
        spam_tokens, spam_list = self.loadData(self.path[1])
        tokens = ham_tokens + spam_tokens
        label = ham_list + spam_list
        self.Tokens = tokens.copy()
        idx = list(range(len(tokens)))
        testSet = []
        testLabel = []
        trainSet = []
        for i in range(int(len(tokens) * ratio)):
            randIdx = int(random.uniform(0, len(tokens)))
            testSet.append(tokens[randIdx])
            testLabel.append(label[randIdx])
            del(idx[randIdx])
            del(tokens[randIdx])
            del(label[randIdx])
        for tok in tokens:
            trainSet.append(self.word2vect(tok))
        return trainSet, label, testSet, testLabel

    def word2vect(self,  input: list):
        self.vocab = self.createVocab()
        vec = [0] * len(self.vocab)
        for word in input:
            vec[self.vocab.index(word)] = 1
        return vec

    def createVocab(self):
        vocab = set([])
        for token in self.Tokens:
            vocab = vocab | set(token)
        return list(vocab)

    def trainClassifier(self, ratio, Laplace=False):
        trainSet, trainLabels, testSet, testLabel = self.split_train_test(ratio)
        self.testSet, self.testLabel = testSet, testLabel
        numLines = len(trainSet)
        numCols = len(trainSet[0])
        p_ham = sum(trainLabels) / float(numLines)
        num_ham, num_spam = np.zeros(numCols), np.zeros(numCols)
        num1 = sum(trainLabels)
        num2 = numLines - num1
        for i in range(numLines):
            if trainLabels[i] == 1:
                num_ham += trainSet[i]
            else:
                num_spam += trainSet[i]
        if Laplace:
            num_ham += 1
            num1 += 2
            num_spam += 1
            num2 += 2
        ham_prob = num_ham / num1
        spam_prob = num_spam / num2
        self.p1, self.p2, self.p1_prior = ham_prob, spam_prob, p_ham
        return ham_prob, spam_prob, p_ham

    def predict(self, input):
        vect = np.array(self.word2vect(input))
        hp = np.log(self.p1)
        sp = np.log(self.p2)
        p_1 = np.sum(vect * hp) + np.log(self.p1_prior)
        p_2 = np.sum(vect * sp) + np.log(1-self.p1_prior)
        if p_1 > p_2:
            return 1
        else:
            return 0

    def testing(self):
        error = 0
        for i, test_vec in enumerate(self.testSet):
            rs = self.predict(test_vec)
            if rs != self.testLabel[i]:
                error += 1
                print('the predict is : {}, true is : {}'.format(rs, self.testLabel[i]))
        print(error)
        print('Accuracy={:.3f}%'.format(100 * (1 -error / len(self.testSet))))


if __name__ == '__main__':
    fileList = ['email/ham', 'email/spam']
    bayes = MyBayes(fileList)
    bayes.trainClassifier(0.3, True)
    bayes.testing()













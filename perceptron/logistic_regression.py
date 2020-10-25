# encoding=utf-8
# @Author: WenDesi
# @Date:   08-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

import time
import math
import random
from mnist import loadDataSet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self,x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        exp_wx = math.exp(wx)

        predict1 = exp_wx / (1 + exp_wx)
        predict0 = 1 / (1 + exp_wx)

        if predict1 > predict0:
            return 1
        else:
            return 0


    def train(self,features, labels):
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index]

            if y == self.predict_(x):
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            # print 'iterater times %d' % time
            time += 1
            correct_count = 0

            wx = sum([self.w[i] * x[i] for i in range(len(self.w))])
            exp_wx = math.exp(wx)

            for i in range(len(self.w)):
                self.w[i] -= self.learning_step * \
                    (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))


    def predict(self,features):
        labels = []

        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))

        return labels

if __name__ == "__main__":
    #print 'Start read data'

    time_1 = time.time()

    train_x, test_x, train_y, test_y = loadDataSet()

    time_2 = time.time()
    #print 'read data cost ',time_2 - time_1,' second','\n'

    #print 'Start training'
    lr = LogisticRegression()
    lr.train(train_x, train_y)

    time_3 = time.time()
    #print 'training cost ',time_3 - time_2,' second','\n'

    #print 'Start predicting'
    test_predict = lr.predict(test_x)
    time_4 = time.time()
    #print 'predicting cost ',time_4 - time_3,' second','\n'

    score = accuracy_score(test_y,test_predict)
    print("The accruacy socre is ", score)

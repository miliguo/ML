# encoding=utf-8
# @Author: WenDesi
# @Date:   08-11-16
# @Email:  wendesi@foxmail.com
# @Last modified by:   WenDesi
# @Last modified time: 08-11-16

import csv
import pandas as pd
from mnist import loadDataSet
from binary_perceptron import Perceptron
from logistic_regression import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':



    test_time = 10

    p = Perceptron()
    lr = LogisticRegression()

    writer = csv.writer(open('result.csv', 'w',newline=''))

    for time in range(test_time):
        #print 'iterater time %d' % time

        train_x, test_x, train_y, test_y = loadDataSet()

        p.train(train_x, train_y)
        lr.train(train_x, train_y)

        p_predict = p.predict(test_x)
        lr_predict = lr.predict(test_x)

        p_score = accuracy_score(test_y, p_predict)
        lr_score = accuracy_score(test_y, lr_predict)

        print('perceptron accruacy score ', p_score)
        print('logistic Regression accruacy score ', lr_score)

        writer.writerow([time,p_score,lr_score])

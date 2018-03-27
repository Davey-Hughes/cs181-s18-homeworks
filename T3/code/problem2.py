# CS 181, Harvard University
# Spring 2016
from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron
from collections import defaultdict
from sklearn import svm
from copy import deepcopy
import time

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.S = set()
        self.alpha = defaultdict(int)

        for i in range(self.numsamples):
            if i % 100 == 0:
                sys.stdout.write("%.2f%%\r" % (100 * i // self.numsamples))
                sys.stdout.flush()

            t = np.random.choice(len(Y))
            x_t = X[t,:]
            y_t = Y[t]
            yhat = 0

            for j in self.S:
                yhat += self.alpha[i] * np.dot(x_t.T, X[j,:])

            if yhat * y_t <= 0:
                self.S.add(t)
                self.alpha[t] = y_t

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            if i % 10 == 0:
                sys.stdout.write("%.2f%%\r" % (100 * i // X.shape[0]))
                sys.stdout.flush()

            yhat = 0
            for j in self.S:
                yhat += self.alpha[j] * np.dot(X[i,:].T, self.X[j,:])

            pred = 1 if yhat >= 0 else -1

            preds.append(pred)

        return np.array(preds)

# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

    def get_yhat(self, x_t):
        yhat = 0
        for i in self.S:
            yhat += self.alpha[i] * np.dot(x_t.T, self.X[i,:])

        return yhat

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.alpha = defaultdict(int)
        self.S = set()

        for i in range(self.numsamples):
            if i % 100 == 0:
                sys.stdout.write("%.2f%%\r" % (100 * i // self.numsamples))
                sys.stdout.flush()

            t = np.random.choice(len(Y))
            x_t = X[t,:]
            y_t = Y[t]

            yhat = self.get_yhat(x_t)

            if yhat * y_t <= self.beta:
                self.S.add(t)
                self.alpha[t] = y_t
                if len(self.S) > self.N:
                    margins = []
                    S_cur = list(deepcopy(self.S))
                    for j in S_cur:
                        m = Y[j] * (self.get_yhat(X[j,:]) - self.alpha[j] * np.dot(X[j,:].T, X[j,:]))
                        margins.append(m)

                    self.S.remove(S_cur[np.argmax(margins)])


    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            if i % 10 == 0:
                sys.stdout.write("%.2f%%\r" % (100 * i // X.shape[0]))
                sys.stdout.flush()

            yhat = 0
            for j in self.S:
                yhat += self.alpha[j] * np.dot(X[i,:].T, self.X[j,:])

            pred = 1 if yhat >= 0 else -1
            preds.append(pred)

        return np.array(preds)



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
val = np.loadtxt("val.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.

# start = time.time()
# k = KernelPerceptron(numsamples)
# k.fit(X,Y)
# print('k train time: ', time.time() - start)
# print('k num suppport vecs: ', len(k.S))
# preds = k.predict(X)
# print('k train accuracy: ', sum(preds == Y) / len(preds))
# preds = k.predict(val[:, :2])
# print('k val accuracy: ', sum(preds == val[:, 2]) / len(preds))
# k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)

# betas = [0, -.1, -.5, .1, .5]
betas = [0]
Ns = [25, 50, 200]

# for b in betas:
for n in Ns:
    # print('beta: ', b)
    print('N: ', n)
    start = time.time()
    bk = BudgetKernelPerceptron(beta, n, numsamples)
    bk.fit(X, Y)
    print('bk train time: ', time.time() - start)
    print('bk num suppport vecs: ', len(bk.S))
    preds = bk.predict(X)
    print('bk train accuracy: ', sum(preds == Y) / len(preds))
    preds = bk.predict(val[:, :2])
    print('bk val accuracy: ', sum(preds == val[:, 2]) / len(preds))
    # bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
    print()

# clf = svm.SVC()
# start = time.time()
# clf.fit(X, Y)
# print('sklearn train time: ', time.time() - start)
# print('sklearn num suppport vecs: ', len(clf.n_support_))
# preds = clf.predict(X)
# print('sklearn train accuracy: ', sum(preds == Y) / len(preds))
# preds = clf.predict(val[:, :2])
# print('sklearn val accuracy: ', sum(preds == val[:, 2]) / len(preds))

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 00:15:02 2018
Regularization L2 on co-centric data
@author: FPTShop
"""

import numpy as np
from matplotlib import pyplot as plt

N = 1000
D = 2
r1 = 5
r2 = 10

theta = 2*np.pi*np.random.random((N//2,1))
a = (np.random.randn(N//2,1)+r1)*np.sin(theta)
b = (np.random.randn(N//2,1)+r1)*np.cos(theta)
X1 = np.concatenate((a,b),axis = 1)

theta = 2*np.pi*np.random.random((N//2,1))
a = (np.random.randn(N//2,1)+r2)*np.sin(theta)
b = (np.random.randn(N//2,1)+r2)*np.cos(theta)
X2 = np.concatenate((a,b),axis = 1)

a = np.ones((N//2,1))
b = np.zeros((N//2,1))
Y = np.concatenate((a,b), axis = 0)

X = np.concatenate((X1, X2), axis = 0)
plt.scatter(X[:,0],X[:,1])
plt.show()

# preprocess
Xp = np.sqrt(np.power(X[:,:1],2)+np.power(X[:,1:2],2))
a = np.ones((N,1))
Xp = np.concatenate((X,Xp,a), axis = 1)

# initialize weight
w = np.random.randn(D+2,1)
lr = 0.001

def sigmoid(p):
    return 1/(1+np.exp(-p))
def forward(p,w):
    return sigmoid(p.dot(w))
def classification_rate(t, y):
    return np.mean(np.round(y)==t)
def cross_entropy(t, y):
    return -np.mean(t*np.log(y)+(1-t)*np.log(1-y)) 

for x in range(5000):
    Yp = forward(Xp,w)
    w += lr*(Xp.T.dot(Y-Yp)-0.01*w)
    if x%100 == 0:
        print(cross_entropy(Y,Yp))
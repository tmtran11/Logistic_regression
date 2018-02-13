# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:53:34 2018

@author: FPTShop
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


all_data = pd.read_csv('ecommerce_data.csv')
all_data = all_data.as_matrix()
data = all_data[:,:-1]
t_data = all_data[:,-1:]
result = all_data[:,-1]
# fix visit_duration
data[:,1] = (data[:,1] - np.mean(data[:,1])) /np.std(data[:,1])
data[:,2] = (data[:,2] - np.mean(data[:,2])) /np.std(data[:,2])
# one-hot coder for 4
row, col = data.shape[0], data.shape[1]
pdata = np.zeros((row, col+3))
pdata[:,:col-1] = data[:,:col-1]
for x in range(row):
    pdata[x,col+int(data[x,-1])-1] = 1
# turn all in binary
X = pdata[result<=1]
Y = t_data[result<=1]

def sigmoid(p):
    return 1/(1+np.exp(-p))
def forward(p,w,b):
    return sigmoid(p.dot(w)+b)
def classification_rate(t, y):
    return np.mean(np.round(y)==t)
def cross_entropy(t, y):
    return -np.mean(t*np.log(y)+(1-t)*np.log(1-y)) #infinite?

data = np.concatenate((X,Y),axis = 1)
np.random.shuffle(data)
train_X = data[0:100,:-1]
train_Y  = data[0:100,-1:]
test_X = data[100:200,:-1]
test_Y = data[100:200,-1:]

w = np.random.randn(col+3,1)
b = 0
lr = 0.001
ce_train = []
ce_test = []

for x in range(10000):
    train_pY = forward(train_X, w, b)
    test_pY = forward(test_X, w, b)
    
    ce_train.append(cross_entropy(train_Y,train_pY))
    ce_test.append(cross_entropy(test_Y,test_pY))
    
    w-= lr*(train_X.T.dot(train_pY-train_Y))
    b-= lr*(train_pY-train_Y)
    
    if x%1000==0:
        print(classification_rate(train_Y, train_pY))

plt.plot(ce_train)
plt.show()



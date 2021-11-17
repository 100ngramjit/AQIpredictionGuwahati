# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 08:18:29 2021

@author: 100ngram
"""

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import pandas as pd
from sklearn import metrics
##from confuse import main

import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('bamunimaidanGHY2.csv')


plt.scatter(dataset["Nox"],dataset["YEAR "])

dataset["weekno"]=dataset.index
dataset.info()
dataset["PM10"].fillna(100, inplace = True)
print(dataset)

X=dataset["weekno"]
Y=dataset["PM10"]
X=np.array(X).reshape((-1, 1))
Y=np.array(Y).reshape((-1,1))
print(X)
print(Y)
X2=dataset["weekno"]
Y2=dataset["YEAR "]

X2=np.array(X).reshape((-1, 1))
Y2=np.array(Y2).reshape((-1,1))

knn = KNeighborsRegressor(
    n_neighbors=10, algorithm='auto', leaf_size=30, weights='uniform')
knn.fit(X, Y)
nn = NearestNeighbors(n_neighbors=10, algorithm='auto', leaf_size=30)
nn.fit(X, Y)

err = metrics.mean_absolute_error(Y2, knn.predict(X2)) * 100
print ("Mean Absolute Error: %f" %err)
##main(Y2, knn.predict(X2))
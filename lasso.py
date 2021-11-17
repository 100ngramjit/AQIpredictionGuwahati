# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:34:40 2021

@author: 100ngram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('bamunimaidanGHY2.csv')


plt.scatter(dataset["Nox"],dataset["YEAR "])

dataset["weekno"]=dataset.index
dataset.info()
dataset["PM10"].fillna(100, inplace = True)
print(dataset)
from sklearn import linear_model
from sklearn import metrics

X=dataset["weekno"]
Y=dataset["PM10"]
X=np.array(X).reshape((-1, 1))
Y=np.array(Y).reshape((-1,1))
print(X)
print(Y)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X,Y)

X2=dataset["weekno"]
Y2=dataset["YEAR "]

X2=np.array(X).reshape((-1, 1))
Y2=np.array(Y2).reshape((-1,1))


preds = model.predict(X2)

err = metrics.mean_absolute_error(Y2, preds)
print ("Mean Absolute Error: %f" % err)
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 19:26:16 2021

@author: 100ngram
"""

import numpy as np
import pandas as pd
from numpy import info


import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model

from keras.layers import Input,Conv1D,MaxPooling2D,Flatten,Dense,Dropout
dataset =pd.read_csv('bamunimaidanGHY3.csv')
dataset["PM10"].fillna(100, inplace = True)
dataset["SO2"].fillna(10, inplace = True)
dataset["Nox"].fillna(20, inplace = True)
dataset["weekno"]=dataset.index
dataset.to_numpy()
dataset.info
print(dataset)
y = dataset.iloc[:, 0:1].values
X = dataset.iloc[:, 6:7].values

model = Sequential()
model.add(Dense(100, input_dim=1, activation='linear'))
model.add(Dense(1,activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1,activation='relu'))
model.add(Dense(25, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mse", optimizer="adam")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=10)
accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
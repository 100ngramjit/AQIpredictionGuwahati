# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:31:02 2020

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
X = X.reshape(X.shape[0], X.shape[1], 1)


model = Sequential()
model.add(Conv1D(32, 1, activation="relu", input_shape=(1, 1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1,activation='linear'))
model.compile(loss="mse", optimizer="adam")
##model.add(Dense(200, input_dim=1, activation='relu'))
##model.add(Dense(100, activation='relu'))
##model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=10)
accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
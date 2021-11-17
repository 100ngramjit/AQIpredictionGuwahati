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
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
dataset =pd.read_csv('bamunimaidanGHY3.csv')
dataset["PM10"].fillna(100, inplace = True)
dataset["SO2"].fillna(10, inplace = True)
dataset["Nox"].fillna(20, inplace = True)
dataset["weekno"]=dataset.index
dataset.to_numpy()
dataset.info
print(dataset)
y = dataset.iloc[:, 2:3].values
X = dataset.iloc[:, 6:7].values


model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10)
# evaluate the keras model
accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:29:09 2021

@author: 100ngram
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
lin = linear_model.LinearRegression()

from sklearn import metrics
dataset1 = pd.read_csv('borgaonGHY.csv')
location1 = "borgaon"
dataset2 = pd.read_csv('pragjyotishGHY.csv')
location2 = "pragjyotish"
dataset3 = pd.read_csv('bamunimaidanGHY3.csv')
location3 = "bamununimaidan"
dataset4 = pd.read_csv('guwahatiuniversityGHY.csv')
location4 = "guwahati university"
dataset5 = pd.read_csv('khanaparaGHY.csv')
location5 = "khanapara"

def plotting(X,y,location,pollutant):
 X_grid = np.arange(min(X), max(X), 0.01)
 X_grid = X_grid.reshape((len(X_grid), 1))
 plt.scatter(X, y, color = 'red')
 plt.plot(X_grid, lin.predict(X_grid), color = 'blue')
 plt.title('linear regression test results for '+location)
 plt.xlabel('days')
 plt.ylabel(pollutant)
 plt.show()

def regression(X,Y,location,pollutant):
 from sklearn.impute import SimpleImputer
 imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
 imputer.fit(Y[:,0:1])
 Y[:,0:1] = imputer.transform(Y[:,0:1])

    
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)  
    
 lin.fit(X_train,y_train)
 Y_result = lin.predict(X_test)

 err = metrics.mean_absolute_error(y_test, Y_result)
 print (pollutant+" "+location+" Mean Absolute Error: %f" % err)
 #main(Y2, preds)
 
 from sklearn.metrics import mean_squared_error
 from math import sqrt
 mse = mean_squared_error(y_test, Y_result)
 rmse = sqrt(mse)
 print(pollutant+" "+location+ " root mean square error %f"%rmse)
 
 plotting(X_test,y_test,location,pollutant)
    

def function(dataset,location):
 Y1 = dataset.iloc[:, 0:1].values
 pollutant1 = "PM10"
 pollutant2 = "SO2"
 pollutant3 = "NOx"
 Y2 = dataset.iloc[:, 1:2].values
 Y3 = dataset.iloc[:, 2:3].values
 X =  dataset.iloc[:, 6:7].values
 regression(X,Y1,location,pollutant1)
 regression(X,Y2,location,pollutant2)
 regression(X,Y3,location,pollutant3)

function(dataset1,location1) 
function(dataset2,location2) 
function(dataset3,location3)
function(dataset4,location4) 
function(dataset5,location5)  
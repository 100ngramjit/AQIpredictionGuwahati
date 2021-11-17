#random forest

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

#average_value = 03


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

month = input("enter month in number ")
year = input("enter year ")
day = input("enter day ")

def plotting(X,y,location,pollutant):
 X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
 X_grid = X_grid.reshape((len(X_grid), 1))
 plt.scatter(X, y, color = 'red')
 plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
 plt.title('decisontree test results for '+location)
 plt.xlabel('days')
 plt.ylabel(pollutant)
 plt.show()

def regression(X,Y,location,pollutant):
 from sklearn.impute import SimpleImputer
 imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
 imputer.fit(Y[:,0:1])
 Y[:,0:1] = imputer.transform(Y[:,0:1])

    
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)  
    
 
 
 regressor.fit(X, Y.ravel())
 
 """score = cross_val_score(regressor,X,Y.ravel(), cv=15 , scoring="neg_mean_absolute_error")
 score = -score
 mean_score = np.mean(score)
 print(pollutant+" "+location+ " absolute mean error %f"%mean_score)"""
 
 from datetime import date
 
 d0 = date(2016, 1, 1)
 d1 = date(int(year),int( month),int( day))
 delta = d1 - d0
 
 serial = (delta.days)/7
 
 print("date is")
 print(d1)
 print("aqi value of pollutant"+pollutant+" in location "+location)
 value = regressor.predict([[serial]])

 print(value[0])


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

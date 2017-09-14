#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:03:39 2017

@author: nissim
"""
# Importing the libraries  

import numpy as  np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing datasets with pandas 

dataset  = pd.read_csv('Data.csv')

# Split features and the dependant variable 

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# To take care of the missing entries in the dataset 
# to replace the missing data by the mean of the values of that column 

from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0  )
# upper bound is excluded runs from 1:2 
imputer.fit(X[:, 1: 3 ])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder 
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0]) 

#Creating dummy variables for the encoded values 
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y) 

# Splitting the data set into Training set and test set 
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)


# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)






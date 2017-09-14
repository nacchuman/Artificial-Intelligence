#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:40:31 2017

@author: nissim
"""

#Data preprocessing 
# Importing the libraries  

import numpy as  np
import matplotlib.pyplot as plt 
import pandas as pd

#Importing datasets with pandas 

dataset  = pd.read_csv('Churn_Modelling.csv')

# Split features and the dependant variable 

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


# Encoding categorical data

from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1]) 

labelencoder_X2 = LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2]) 

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#Avioiding the dummy variable trap by dropping one column from the encoded variables 
X = X[:, 1:]
 

# Splitting the data set into Training set and test set 
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)


# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Making the ann
import keras 
from keras.models import Sequential
from keras.layers import Dense 

#Initialising the ann
classifier = Sequential()

#Adding the input layer and first hidden layer 

classifier.add(Dense(output_dim =6 , init='uniform' , activation ='relu' , input_dim = 11) )

#Adding the second layer 
classifier.add(Dense(output_dim =6 , init='uniform' , activation ='relu' ) )

#Adding the output layer  
# if the dependent variable has more than two categories use softmax as activation function 

classifier.add(Dense(output_dim =1 , init='uniform' , activation ='sigmoid' ) )

#Compiling the ANN
# since sigmoid is used in the output layer we will use a sigmoid loss function 
#That is a logarithmic loss function 
#if the dependent variable has more than one outcomes the loss function is called
#categorical_cross_entropy 
#Since we have a binanry output we will have binary_crossentropy

classifier.compile(optimizer= 'adam' , loss ='binary_crossentropy' , metrics = ['accuracy'])


#Fitting the ANN to the training set 
classifier.fit(X_train,Y_train,  batch_size = 10 , nb_epoch = 100   )

#MAking the predictions and evaluating the model 
#Predicting the test set 
y_pred = classifier.predict(X_test)
#if y pred is larger than 0.5 it gives a true and gives false otherwise
y_pred = (y_pred > 0.5 )
#Making the confusion matrix 

from sklearn.metrics import confusion_matrix

cm  = confusion_matrix(Y_test, y_pred)
#Accuracy


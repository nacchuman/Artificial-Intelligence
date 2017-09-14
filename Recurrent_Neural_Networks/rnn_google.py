#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 04:26:07 2017

@author: nissim
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing the dataset
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

#Feature scling standardisation and normalisation 
#normalisationgives better results

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set  = sc.fit_transform(training_set)

#Getting the inputs and the outputs 
X_train = training_set[0:1257]
Y_train = training_set[1:1258]

#Repashing 
#We need to reshape because recurrent layers in keras  expects (check documentation)
#keras expects the following shape  (observations, timestamp,features)
X_train = np.reshape(X_train, (1257, 1,1))

#Building the RNN

#Importing keras libs

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#Initialising the RNN

regressor = Sequential()
#Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4 , activation = 'sigmoid' , input_shape = (None , 1)))
#Adding the output layer
regressor.add(Dense(units=1))
#Compiling the RNN use rms prop optimizer we are useing adam here 

regressor.compile(optimizer ='adam', loss='mean_squared_error')
#Fitting the RNN To the training set 

regressor.fit(X_train,Y_train,batch_size=32, epochs=200)
#Part 3 - making the predictions and visualising the resluts

#Getting the predicted stock prices of 2017

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price 
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price,color = 'red', label = 'Real Google stock price')

plt.plot(predicted_stock_price,color = 'blue', label = 'Predicted Google stock price')
plt.title('Google stock prices prediction')
plt.xlabel('Time')
plt.ylabel('Google stock prices')
plt.legend()
plt.show()

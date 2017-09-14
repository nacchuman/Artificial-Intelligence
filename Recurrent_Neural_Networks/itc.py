#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 05:38:32 2017

@author: nissim
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing the dataset

training_set = pd.read_csv('ITC.NS.csv')
# sets null values in the open column to nan
training_set.loc[training_set.Open == 'null'] = np.nan
#pads the nan value with a legitimate value berfore it
training_set = training_set.fillna(method='pad')
training_set = training_set.iloc[:5400,1:2].values

#Feature scling standardisation and normalisation 
#normalisationgives better results

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set  = sc.fit_transform(training_set)

#Getting the inputs and the outputs 
X_train = training_set[0:5399]
Y_train = training_set[1:5400]

#Repashing 
#We need to reshape because recurrent layers in keras  expects (check documentation)
#keras expects the following shape  (observations, timestamp,features)
X_train = np.reshape(X_train, (5399, 1,1))

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

test_set = pd.read_csv('ITC.NS.csv')
test_set.loc[test_set.Open == 'null'] = np.nan
test_set = test_set.fillna(method='pad')

real_stock_price = test_set.iloc[5401:,1:2].values
real_stock_price = sc.transform(real_stock_price)
real_stock_price = sc.inverse_transform(real_stock_price)


inputs = real_stock_price 
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (86, 1,1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price,color = 'red', label = 'Real ITC stock price')

plt.plot(predicted_stock_price,color = 'blue', label = 'Predicted ITC stock price')
plt.title('ITC stock prices prediction')
plt.xlabel('Time')
plt.ylabel('ITC stock prices')
plt.legend()
plt.show()


from alpha_vantage.timeseries import TimeSeries
import sys

def stockchart(symbol):
    ts = TimeSeries(key='UYHAS3N088KVTKRZ', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol,interval='1min', outputsize='full')
    print (data)
    data['close'].plot()
    plt.title('Stock chart')
    plt.show()
    
symbol='ITC.NS'
stockchart(symbol)
import csv


def stockMonthly(symbol):
    ts = TimeSeries(key='UYHAS3N088KVTKRZ', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol,datatype=csv)
    print (data)
    
stockMonthly(symbol)
    

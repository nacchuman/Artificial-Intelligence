#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:49:23 2017

@author: nissim
"""

import numpy as np 
import tensorflow as tf 

#Model parameters 

W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
#Model input and output 
x = tf.placeholder(tf.float32)
linear_model = W * x + b 
y = tf.placeholder(tf.float32)

#loss 
loss = tf.reduce_sum(tf.square(linear_model - y )) # Sum of squares 

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
#Training data 
x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]
 
#Training loop 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train , {x:x_train , y:y_train})
    
#Evaluate training accuracy 
curr_W ,curr_b , curr_loss = sess.run([W , b , loss], {x:x_train,y:y_train})
print("W : %s , b: %s , loss: %s"%(curr_W ,curr_b , curr_loss ))
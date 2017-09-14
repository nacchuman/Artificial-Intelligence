#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:33:04 2017

@author: nissim
"""

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x =tf.placeholder(tf.float32 , shape=[None, 784])
y_ = tf.placeholder(tf.float32 , shape = [None , 10])


W = tf.Variable(tf.zeros([784 , 10]))
b= tf.Variable (tf.zeros([10]))


y = tf.matmul(x, W) + b 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_ , logits= y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
temp_w = sess.run(W)
correct_prediction = tf.equal(tf.argmax(y,1) , tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for _ in range(10000):
    batch = mnist.train.next_batch(100)
    
    print("Training on ", _ ,"batch ", accuracy.eval(feed_dict={x : batch[0] , y_:batch[1]}))

    train_step.run(feed_dict={x : batch[0] , y_:batch[1]})
    
print("accuracy over test set ")
print(accuracy.eval(feed_dict={ x:mnist.test.images , y_:mnist.test.labels}))

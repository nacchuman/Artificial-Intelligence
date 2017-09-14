#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:58:37 2017

@author: nissim
"""

"""
The central unit of data in TensorFlow is the tensor. 
A tensor consists of a set of primitive values shaped into an array of any number of dimensions.
A tensor's rank is its number of dimensions. Here are some examples of tensors:
"""
import tensorflow as tf 
#, it takes no inputs, and it outputs a value it stores internally.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# Actually evaluate the nodes, we must run the computational graph within a session.
# A session encapsulates the control and state of the TensorFlow runtime.

sess = tf.Session()
print(sess.run([node1, node2 ]))

node3 = tf.add(node1, node2)
print("Node 3", node3)
print("sess.run(node3)",sess.run(node3))


# A graph can be parameterized to accept external inputs, known as placeholders.
# A placeholder is a promise to provide a value later

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

#We can make the computational graph more complex by adding another operation. For example,

add_and_triple = adder_node * 3 
print(sess.run(add_and_triple , {a:3,b:4.5}))

#Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
#Constants are initialized when you call tf.constant, and their value can never change.
# By contrast, variables are not initialized when you call tf.Variable. 
#To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
    
init = tf.global_variables_initializer()
sess.run(init)
# fires the linear model for all the values in x 
print(sess.run(linear_model, {x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model  - y )
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b , [1.])
sess.run( [fixW, fixb ] )
print(sess.run(loss, {x:[1,2,3,4], y :[0, -1 , -2 , -3 ]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))



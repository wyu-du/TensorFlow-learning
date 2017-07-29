# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 09:06:18 2017

@author: think
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate training data
np.random.seed(1)
def f(x,a,b):
    n=train_X.size
    vals=np.zeros((1,n))
    for i in range(0,n):
        ax=np.multiply(a,x.item(i))
        val=np.add(ax,b)
        vals[0,i]=val
    return vals

Wref=0.7
bref=-1
n=20
noise_var=0.001
train_X=np.random.random((1,n))
ref_Y=f(train_X,Wref,bref)
train_Y=ref_Y+np.sqrt(noise_var)*np.random.randn(1,n)
n_samples=train_X.size
print (" Type of 'train_X' is ", type(train_X))
print (" Shape of 'train_X' is %s" % (train_X.shape,))
print (" Type of 'train_Y' is ", type(train_Y))
print (" Shape of 'train_Y' is %s" % (train_Y.shape,))

# Plot
plt.figure(1)
plt.plot(train_X[0,:],ref_Y[0,:],'ro',label='Original Data')
plt.plot(train_X[0,:],train_Y[0,:],'bo',label='Training Data')
plt.axis('equal')
plt.legend(loc='lower right')

# Prepare for linear regression
# parameters
training_epochs=2000
display_step=50
# set tensorflow graph
X=tf.placeholder(tf.float32,name='input')
Y=tf.placeholder(tf.float32,name='output')
W=tf.Variable(np.random.randn(),name='weight')
b=tf.Variable(np.random.randn(),name='bias')
# construct a model
activation=tf.add(tf.multiply(X,W),b)
# define error measure and optimizer
learning_rate=0.01
cost=tf.reduce_mean(tf.pow(activation-Y,2))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# initializer
init=tf.initialize_all_variables()
# run
sess=tf.Session()
# initialize
sess.run(init)
for epoch in range(training_epochs):
    for (x,y) in zip(train_X[0,:],train_Y[0,:]):
        sess.run(optimizer,feed_dict={X:x,Y:y})
    #check cost
    if epoch%display_step==0:
        costval=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
        print('Epoch:','%04d'%(epoch+1),'cost=','{:.5f}'.format(costval))
        Wtemp=sess.run(W)
        btemp=sess.run(b)
        print('Wtemp is','{:.4f}'.format(Wtemp),'btemp is','{:.4f}'.format(btemp))
        print (" Wref is", "{:.4f}".format(Wref), "bref is", "{:.4f}".format(bref))
# final W and b
Wopt=sess.run(W)
bopt=sess.run(b)
fopt=f(train_X,Wopt,bopt)

# plot results
plt.figure(2)
plt.plot(train_X[0,:],ref_Y[0,:],'ro',label='Original Data')
plt.plot(train_X[0,:],train_Y[0,:],'bo',label='Training Data')
plt.plot(train_X[0,:],fopt[0,:],'k-',label='Fitted Line')
plt.axis('equal')
plt.legend(loc='lower right')

        
        
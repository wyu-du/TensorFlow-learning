# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 10:01:40 2017

@author: think
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('data/',one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST loaded")

# Create tensor graph for logistic regression
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float',[None,10])  #None is for infinite
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
# logistic regression model
'''
WEIGHT_DECAY_FACTOR=1  #0.000001
l2_loss=tf.add_n([tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()])
'''
actv=tf.nn.softmax(tf.matmul(x,W)+b)
# cost function
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))
'''
cost=cost+WEIGHT_DECAY_FACTOR*l2_loss
'''
# optimizer
learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Prediction and accuracy
# prediction
pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
# accuracy
accr=tf.reduce_mean(tf.cast(pred,'float'))
# initializer
init=tf.initialize_all_variables()

# Train model
training_epochs=50
batch_size=100
display_step=5
# session
sess=tf.Session()
sess.run(init)
# mini-batch learning
for epoch in range(training_epochs):
    avg_cost=0.
    num_batch=int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
        feeds={x:batch_xs,y:batch_ys}
        avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch
    #display
    if epoch%display_step==0:
        feeds_train={x:batch_xs,y:batch_ys}
        feeds_test={x:mnist.test.images,y:mnist.test.labels}
        train_acc=sess.run(accr,feed_dict=feeds_train)
        test_acc=sess.run(accr,feed_dict=feeds_test)
        print("Epoch:%03d/%03d cost: %.9f train_aac: %.3f test_acc: %.3f"
              %(epoch,training_epochs,avg_cost,train_acc,test_acc))
print("Done!")
sess.close()
print("Session closed.")
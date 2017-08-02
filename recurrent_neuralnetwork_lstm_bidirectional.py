# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:38:08 2017

@author: think
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
import numpy as np

mnist=input_data.read_data_sets('data/',one_hot=True)

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
lr=0.001
training_iters=100000
batch_size=128
display_step=100

# Network params
n_input=28
n_steps=28
n_hidden=128
n_classes=10

# tf graph input
x=tf.placeholder('float',[None,n_steps,n_input])
y=tf.placeholder('float',[None,n_classes])

# Define weights
weights={
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out':tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
        }
biases={
        'out':tf.Variable(tf.random_normal([n_classes]))
        }

def BiRNN(x,weights,biases):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x=tf.unstack(x,n_steps,1)
    # Define lstm cells with tf
    # Forward direction cell
    lstm_fw_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell=rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # Get lstm cell output
    try:
        outputs,_,_=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    except Exception:
        outputs=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    output=tf.matmul(outputs[-1],weights['out'])+biases['out']
    return output

pred=BiRNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# Evaluate model
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x=batch_x.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter "+str(step*batch_size)+", Minibatch Loss="+"{:.6f}".format(loss)+", Training Accuracy= "+"{:.5f}".format(acc))
        step+=1
    print("Optimization Finished!")
    # Calculate accuracy for 128 mnist test images
    test_len=128
    test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
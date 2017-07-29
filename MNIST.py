# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:03:22 2017

@author: think
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print("Download and Extract MNIST dataset")
mnist=input_data.read_data_sets('data/',one_hot=True)
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))

print ("What does the data of MNIST look like?")
trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))

print ("How does the training data look like?")
nsample=5
randidx=np.random.randint(trainimg.shape[0],size=nsample)
for i in randidx:
    curr_img=np.reshape(trainimg[i,:],(28,28))  #28*28 matrix
    curr_label=np.argmax(trainlabel[i,:])
    plt.matshow(curr_img,cmap=plt.get_cmap('gray'))
    plt.title(""+str(i)+"th Training Data"
                +" Label is "+str(curr_label))
    print(""+str(i)+"th Training Data"
                +" Label is "+str(curr_label))

print ("Batch Learning? ")
batch_size=100
batch_xs,batch_ys=mnist.train.next_batch(batch_size)
print ("shape of 'batch_xs' is %s" % (batch_xs.shape,))
print ("shape of 'batch_ys' is %s" % (batch_ys.shape,))

print ("5. Get Random Batch with 'np.random.randint'")
randidx=np.random.randint(trainimg.shape[0],size=batch_size)
batch_xs2=trainimg[randidx,:]
batch_ys2=trainlabel[randidx,:]
print ("shape of 'batch_xs2' is %s" % (batch_xs2.shape,))
print ("shape of 'batch_ys2' is %s" % (batch_ys2.shape,))

print(randidx)
    
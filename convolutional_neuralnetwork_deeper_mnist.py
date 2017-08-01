# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:37:39 2017

@author: think
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels

# Define CNN
n_input=784
n_output=10
weights={
        #weights are [height, width, in_channels, out_channels]
        'wc1':tf.Variable(tf.truncated_normal([3,3,1,64],stddev=0.1)),
        'wc2':tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.1)),
        'wd1':tf.Variable(tf.truncated_normal([7*7*128,1024],stddev=0.1)),
        'wd2':tf.Variable(tf.truncated_normal([1024,n_output],stddev=0.1))
}
biases={
        'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
        'bc2':tf.Variable(tf.random_normal([128],stddev=0.1)),
        'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
        'bd2':tf.Variable(tf.random_normal([n_output],stddev=0.1))
}
def conv_basic(_input,_w,_b,_keepratio):
    # input 
    _input_r=tf.reshape(_input,shape=[-1,28,28,1])
    # conv layer 1
    _conv1=tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')
    _mean,_var=tf.nn.moments(_conv1,[0,1,2])
    _conv1=tf.nn.batch_normalization(_conv1,_mean,_var,0,1,0.0001)
    _conv1=tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))
    _pool1=tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr1=tf.nn.dropout(_pool1,_keepratio)
    # conv layer 2
    _conv2=tf.nn.conv2d(_pool_dr1,_w['wc2'],strides=[1,1,1,1],padding='SAME')
    _mean,_var=tf.nn.moments(_conv2,[0,1,2])
    _conv2=tf.nn.batch_normalization(_conv2,_mean,_var,0,1,0.0001)
    _conv2=tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    _pool2=tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr2=tf.nn.dropout(_pool2,_keepratio)
    # vectorize
    _dense1=tf.reshape(_pool_dr2,[-1,_w['wd1'].get_shape().as_list()[0]])
    # fully connected layer 1
    _fc1=tf.nn.relu(tf.add(tf.matmul(_dense1,_w['wd1']),_b['bd1']))
    _fc_dr1=tf.nn.dropout(_fc1,_keepratio)
    # fully connected layer 2
    _out=tf.add(tf.matmul(_fc_dr1,_w['wd2']),_b['bd2'])
    # return
    out={
            'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
            'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
            }
    return out

# Define computational graph
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_output])
keepratio=tf.placeholder(tf.float32)

_pred=conv_basic(x,weights,biases,keepratio)['out']
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))
optm=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr=tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(_corr,tf.float32))
init=tf.global_variables_initializer()

save_step=1
saver=tf.train.Saver(max_to_keep=3)

# Optimize
do_train=1
sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

training_epochs=15
batch_size=100
display_step=1
if do_train==1:
    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.7})
            avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.0})/total_batch
        if epoch%display_step==0:
            train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1.0})
            test_acc=sess.run(accr,feed_dict={x:testimg,y:testlabel,keepratio:1.0})
            print("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
                  %(epoch,training_epochs,avg_cost,train_acc,test_acc))
        if epoch%save_step==0:
            saver.save(sess,"nets/cnn_mnist_basic.ckpt-"+str(epoch))
            
if do_train==0:
    epoch=training_epochs-1
    saver.restore(sess,"nets/cnn_mnist_basic.ckpt-"+str(epoch))
    
test_acc=sess.run(accr,feed_dict={x:testimg,y:testlabel,keepratio:1.0})
print("TEST ACCURACY: %.3f"%(test_acc))

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:27:26 2017

@author: think
"""

from __future__ import print_function
import tensorflow as tf
import random

class ToySequenceData(object):
    '''
    Generate sequence of data with dynamic length.
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    '''
    def __init__(self, n_sampmles=1000, max_seq_len=20, min_seq_len=3, max_value=1000):
        self.data=[]
        self.lables=[]
        self.seqlen=[]
        for i in range(n_sampmles):
            l=random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(l)
            if random.random()<0.5:
                rand_start=random.randint(0, max_value-l)
                s=[ [float(i)/max_value] for i in range(rand_start, rand_start+l)]
                s+=[ [0.] for i in range(max_seq_len-l)]
                self.data.append(s)
                self.lables.append([1.,0.])
            else:
                s=[ [float(random.randint(0, max_value)/max_value)] for i in range(l)]
                s+=[ [0.] for i in range(max_seq_len-l)]
                self.data.append(s)
                self.lables.append([0.,1.])
        self.batch_id=0
        
    def next(self, batch_size):
        if self.batch_id==len(self.data):
            self.batch_id=0
        batch_data=self.data[self.batch_id:min(self.batch_id+batch_size,len(self.data))]
        batch_labels=self.lables[self.batch_id:min(self.batch_id+batch_size,len(self.data))]
        batch_seqlen=self.seqlen[self.batch_id:min(self.batch_id+batch_size,len(self.data))]
        self.batch_id=min(self.batch_id+batch_size,len(self.data))
        return batch_data,batch_labels,batch_seqlen
    
lr=0.01
training_iters=1000000
batch_size=128
display_step=100

seq_max_len=20
n_hidden=64
n_classes=2

trainset=ToySequenceData(n_sampmles=1000,max_seq_len=seq_max_len)
testset=ToySequenceData(n_sampmles=500,max_seq_len=seq_max_len)

x=tf.placeholder('float',[None,seq_max_len,1])
y=tf.placeholder('float',[None,n_classes])
seqlen=tf.placeholder(tf.int32,[None])

weights={
        'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
        }
biases={
        'out':tf.Variable(tf.random_normal([n_classes]))
        }

def dynamicRNN(x, seqlen, weights, biases):
    # Current data input shape: (batch_size, n_steps, n_input)
    # Change data shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x=tf.unstack(x, seq_max_len, 1)
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic calculation.
    outputs, states=tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    # 'outputs' is a list of output at every timestep
    outputs=tf.stack(outputs)
    # change back the 'outputs' dimension to [batch_size, n_step, n_input]
    outputs=tf.transpose(outputs,[1,0,2])
    
    batch_size=tf.shape(outputs)[0]
    index=tf.range(0,batch_size)*seq_max_len+(seqlen-1)
    outputs=tf.gather(tf.reshape(outputs,[-1,n_hidden]),index)
    
    return tf.matmul(outputs,weights['out'])+biases['out']

pred=dynamicRNN(x, seqlen, weights, biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<training_iters:
        batch_x, batch_y, batch_seqlen=trainset.next(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})
            loss=sess.run(cost,feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= "+"{:.6f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))
        step+=1
    print("Optimization Finished!")
    
    test_data=testset.data
    test_label=testset.lables
    test_seqlen=testset.seqlen
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_data, y:test_label, seqlen:test_seqlen}))
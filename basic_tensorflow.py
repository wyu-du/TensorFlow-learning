# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:29:42 2017

@author: think
"""

import numpy as np
import tensorflow as tf

#open session
sess=tf.Session()

#tf constant类型
def print_tf(x):
    print("TYPE IS\n %s"%(type(x)))
    print("VALUE IS\n %s"%(x))
hello=tf.constant("HELLO. IT'S ME.")
print_tf(hello)

#运行constant
hello_out=sess.run(hello)
print_tf(hello_out)

#其他类型的constant
a=tf.constant(1.5)
b=tf.constant(2.5)
print_tf(a)
a_out=sess.run(a)
print_tf(a_out)

#operators
a_plus_b=tf.add(a,b)
a_plus_b_out=sess.run(a_plus_b)

a_mul_b=tf.multiply(a,b)
a_mul_b_out=sess.run(a_mul_b)

#variables（所有的变量都只有在被初始化之后才能被有值）
weight=tf.Variable(tf.random_normal([5,2],stddev=0.1))
init=tf.initialize_all_variables()
sess.run(init)
weight_out=sess.run(weight)
print_tf(weight_out)

#placeholders(None表示输入的行数不确定，5表示输入5列)
x=tf.placeholder(tf.float32,[None,5])
#operation with variables and placeholders
oper=tf.matmul(x,weight)
data=np.random.rand(1,5)
#placeholder为占位符，通过feed_dict将值传给x
oper_out=sess.run(oper,feed_dict={x:data})
print_tf(oper_out)
data=np.random.rand(2,5)
#每次传递的值不同，运行operation的结果就不同
oper_out=sess.run(oper,feed_dict={x:data})
print_tf(oper_out)
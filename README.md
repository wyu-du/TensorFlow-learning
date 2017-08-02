Study records of the tensorflow [#1](https://github.com/sjchoi86/Tensorflow-101) [#2](https://github.com/aymericdamien/TensorFlow-Examples)
======

## Machine Learing Basics with TensorFlow
1. Linear Regression <br>
![](https://github.com/ddddwy/TensorFlow-learning/raw/master/images/linear_regression1.png)<br>
![](https://github.com/ddddwy/TensorFlow-learning/raw/master/images/linear_regression2.png)<br>

2. Logistic Regression with MNIST<br>
* 1-layer nueral network, learningrate=0.01，training_epoch=50 :<br>
		cost: 0.306350609 train_aac: 0.900 test_acc: 0.918<br>
* 1-layer nueral network, learningrate=0.01，training_epoch=50, learningrate_decay=0.000001, add L2 regularization :<br>
		cost: 1.909791144 train_aac: 0.790 test_acc: 0.791<br>

		
## Neural Networks	
1. Multi-Layer Perceptron (MLP)
* 2-layer nerual network, h1(784x256), h2(256x128) ：<br>
		cost: 0.002916916 train_aac: 1.000 test_acc: 0.980<br>

* 3-layer nerual network, h1(784x512), h2(512x512), h3(512x256), dropout_keep_prob=0.6 ：<br>
		cost: 0.003112582 train_aac: 1.000 test_acc: 0.981<br>

		
2. Convolutional Neural Network (CNN)
* 1 convolutional layer, 1 fully connected layer :<br>
		cost: 0.023908507 train_aac: 1.000  test_acc: 0.984<br>
		Size of 'input_r' is (1, 28, 28, 1)<br>
		convolution: Size of 'conv1' is (1, 28, 28, 64), size of 'wc1' is (3, 3, 1, 64)<br>
		add bias: Size of 'conv2' is (1, 28, 28, 64)<br>
		pass relu: Size of 'conv3' is (1, 28, 28, 64)<br>
		max pooling: Size of 'pool' is (1, 14, 14, 64)<br>
		fully connected layer: Size of 'dense' is (1, 12544)<br>
		Size of 'output' is (1, 10)<br>

* 2 convolutional layer, 2 fully connected layer, keep_ratio=0.7 :<br>
		cost: 0.026513918 train_acc: 0.970 test_acc: 0.987<br>
		test accuracy is higher than one-layer, but takes much much more time.<br>

		
3. Recurrent Neural Network (RNN)
* A Recurrent Neural Network (LSTM): <br>
		1 simple lstm cell <br>
		Test Accuracy: 0.984375<br>

* A Bidirectional Recurrent Neural Network (LSTM):<br>
		2 lstm cells: forward dircetion cell + backward direction cell <br>
		Test Accuracy: 1.0<br>

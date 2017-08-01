Study records of the [tensorflow tutorial](https://github.com/sjchoi86/Tensorflow-101)
======

## Machine Learing Basics with TensorFlow
1. Linear Regression <br>
![](https://github.com/ddddwy/TensorFlow-learning/raw/master/images/linear_regression1.png)<br>
![](https://github.com/ddddwy/TensorFlow-learning/raw/master/images/linear_regression2.png)<br>

2. Logistic Regression with MNIST<br>
* 1-layer nueral network(784*10), learningrate=0.01，training_epoch=50:<br>
		Epoch:000/050 cost: 1.176854430 train_aac: 0.850 test_acc: 0.851<br>
		Epoch:005/050 cost: 0.440934456 train_aac: 0.960 test_acc: 0.895<br>
		Epoch:010/050 cost: 0.383406833 train_aac: 0.900 test_acc: 0.905<br>
		Epoch:015/050 cost: 0.357249970 train_aac: 0.910 test_acc: 0.909<br>
		Epoch:020/050 cost: 0.341468048 train_aac: 0.870 test_acc: 0.912<br>
		Epoch:025/050 cost: 0.330536755 train_aac: 0.910 test_acc: 0.914<br>
		Epoch:030/050 cost: 0.322361949 train_aac: 0.930 test_acc: 0.915<br>
		Epoch:035/050 cost: 0.315932722 train_aac: 0.860 test_acc: 0.916<br>
		Epoch:040/050 cost: 0.310719569 train_aac: 0.930 test_acc: 0.918<br>
		Epoch:045/050 cost: 0.306350609 train_aac: 0.900 test_acc: 0.918<br>
* 1-layer nueral network(784*10), learningrate=0.01，training_epoch=50, learningrate_decay=0.000001, add L2 regularization:<br>
		Epoch:000/050 cost: 1.955112785 train_aac: 0.780 test_acc: 0.785<br>
		Epoch:005/050 cost: 1.909684022 train_aac: 0.790 test_acc: 0.794<br>
		Epoch:010/050 cost: 1.909698583 train_aac: 0.900 test_acc: 0.786<br>
		Epoch:015/050 cost: 1.909712023 train_aac: 0.870 test_acc: 0.806<br>
		Epoch:020/050 cost: 1.909823188 train_aac: 0.770 test_acc: 0.792<br>
		Epoch:025/050 cost: 1.909710920 train_aac: 0.750 test_acc: 0.795<br>
		Epoch:030/050 cost: 1.909693537 train_aac: 0.810 test_acc: 0.796<br>
		Epoch:035/050 cost: 1.909767549 train_aac: 0.760 test_acc: 0.800<br>
		Epoch:040/050 cost: 1.910010241 train_aac: 0.860 test_acc: 0.802<br>
		Epoch:045/050 cost: 1.909791144 train_aac: 0.790 test_acc: 0.791<br>

## Multi-Layer Perceptron (MLP)
1. Simple MNIST<br>
* 2-layer nerual network, h1(784*256), h2(256*128)：<br>
		Epoch: 003/020 cost: 0.119174641<br>
		TRAIN ACCURACY: 0.950<br>
		TEST ACCURACY: 0.963<br>
		Epoch: 007/020 cost: 0.047656634<br>
		TRAIN ACCURACY: 0.980<br>
		TEST ACCURACY: 0.977<br>
		Epoch: 011/020 cost: 0.018130043<br>
		TRAIN ACCURACY: 0.980<br>
		TEST ACCURACY: 0.978<br>
		Epoch: 015/020 cost: 0.006936280<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.979<br>
		Epoch: 019/020 cost: 0.002916916<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.980<br>

2. Deeper MNIST<br>
* 3-layer nerual network, h1(784*512), h2(512*512), h3(512*256), dropout_keep_prob=0.6：<br>
		Epoch: 003/020 cost: 0.032308363<br>
		TRAIN ACCURACY: 0.980<br>
		TEST ACCURACY: 0.977<br>
		Epoch: 007/020 cost: 0.010782411<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.981<br>
		Epoch: 011/020 cost: 0.004934132<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.980<br>
		Epoch: 015/020 cost: 0.002741719<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.980<br>
		Epoch: 019/020 cost: 0.003112582<br>
		TRAIN ACCURACY: 1.000<br>
		TEST ACCURACY: 0.981<br>
		
## Convolutional Neural Network (CNN)
1. Simple MNIST<br>
* one convolutional layer(3*3*1*64), one densen layer(14*14*64), strides(1*1*1*1), padding='SAME', max pooling:<br>
		Training accuracy: 1.000<br>
		Test accuracy: 0.984<br>
		Size of 'input_r' is (1, 28, 28, 1)<br>
		convolution: Size of 'conv1' is (1, 28, 28, 64), size of 'wc1' is (3, 3, 1, 64)<br>
		add bias: Size of 'conv2' is (1, 28, 28, 64)<br>
		pass relu: Size of 'conv3' is (1, 28, 28, 64)<br>
		max pooling: Size of 'pool' is (1, 14, 14, 64)<br>
		fully connected layer: Size of 'dense' is (1, 12544)<br>
		output: Size of 'out' is (1, 10)<br>


<<<<<<< HEAD
Records of studying the [tensorflow tutorial](https://github.com/sjchoi86/Tensorflow-101)
<br>
## Machine Learing Basics with TensorFlow
1. Linear Regression <br>
![](https://github.com/ddddwy/tensorflow_study/raw/master/images/linear_regression1.png)
![](https://github.com/ddddwy/tensorflow_study/raw/master/images/linear_regression2.png)

2. Logistic Regression with MNIST
* 单层神经网络（784*10）训练MNIST数据集，学习率为0.01，训练迭代次数50次：
		Epoch:000/050 cost: 1.176854430 train_aac: 0.850 test_acc: 0.851
		Epoch:005/050 cost: 0.440934456 train_aac: 0.960 test_acc: 0.895
		Epoch:010/050 cost: 0.383406833 train_aac: 0.900 test_acc: 0.905
		Epoch:015/050 cost: 0.357249970 train_aac: 0.910 test_acc: 0.909
		Epoch:020/050 cost: 0.341468048 train_aac: 0.870 test_acc: 0.912
		Epoch:025/050 cost: 0.330536755 train_aac: 0.910 test_acc: 0.914
		Epoch:030/050 cost: 0.322361949 train_aac: 0.930 test_acc: 0.915
		Epoch:035/050 cost: 0.315932722 train_aac: 0.860 test_acc: 0.916
		Epoch:040/050 cost: 0.310719569 train_aac: 0.930 test_acc: 0.918
		Epoch:045/050 cost: 0.306350609 train_aac: 0.900 test_acc: 0.918
* 单层神经网络（784*10）训练MNIST数据集，学习率为0.01，训练迭代次数50次，学习率衰减为0.000001，加入L2正则：
		Epoch:000/050 cost: 1.955112785 train_aac: 0.780 test_acc: 0.785
		Epoch:005/050 cost: 1.909684022 train_aac: 0.790 test_acc: 0.794
		Epoch:010/050 cost: 1.909698583 train_aac: 0.900 test_acc: 0.786
		Epoch:015/050 cost: 1.909712023 train_aac: 0.870 test_acc: 0.806
		Epoch:020/050 cost: 1.909823188 train_aac: 0.770 test_acc: 0.792
		Epoch:025/050 cost: 1.909710920 train_aac: 0.750 test_acc: 0.795
		Epoch:030/050 cost: 1.909693537 train_aac: 0.810 test_acc: 0.796
		Epoch:035/050 cost: 1.909767549 train_aac: 0.760 test_acc: 0.800
		Epoch:040/050 cost: 1.910010241 train_aac: 0.860 test_acc: 0.802
		Epoch:045/050 cost: 1.909791144 train_aac: 0.790 test_acc: 0.791

## Multi-Layer Perceptron (MLP)
1. Simple MNIST
* 两层神经网络，h1的结构为784*256，h2结构为256*128，输出10个label：
		Epoch: 003/020 cost: 0.119174641
		TRAIN ACCURACY: 0.950
		TEST ACCURACY: 0.963
		Epoch: 007/020 cost: 0.047656634
		TRAIN ACCURACY: 0.980
		TEST ACCURACY: 0.977
		Epoch: 011/020 cost: 0.018130043
		TRAIN ACCURACY: 0.980
		TEST ACCURACY: 0.978
		Epoch: 015/020 cost: 0.006936280
		TRAIN ACCURACY: 1.000
		TEST ACCURACY: 0.979
		Epoch: 019/020 cost: 0.002916916
		TRAIN ACCURACY: 1.000
		TEST ACCURACY: 0.980
||||||| merged common ancestors
=======
# tensorflow_study
study the tensorflow tutorial from https://github.com/sjchoi86/Tensorflow-101
>>>>>>> 47cb0a055c849830dcc54f483e9cd3bade112631

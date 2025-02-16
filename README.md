# Machine Learning/Deep Learning
This repository consists of **two main parts**: a simple **deep learning framework** ([npmod](https://github.com/CryAndRRich/npmod/tree/main/npmod)) and some basic **machine learning models** ([modelML](https://github.com/CryAndRRich/npmod/tree/main/modelML)). Since this is for learning purposes, everything is built **from scratch** using NumPy and PyTorch (mostly NumPy)

## npmod: NumPy module
* Activations:
  * ReLU, leaky ReLU
  * Sigmoid
  * Tanh
  * Softmax
* Layers:
  * Linear
  * Dropout
  * Flatten
* Sequential
* Losses:
  * MAE
  * MSE
  * MALE
  * RSquared
  * MAPE, wMAPE
  * SmoothL1
  * CE, BCE
  * KLDiv
* Optimizers:
  * GD, SGD
  * AdaGrad
  * RMSprop
  * ADAM

## modelML:
Every model is built using NumPy or PyTorch or both

### Machine Learning:
* Linear Regression (NumPy and PyTorch)
* Logistic Regression (NumPy and PyTorch)
* Perceptron Learning (NumPy and PyTorch)
* K-Nearest Neighbors (PyTorch)
* Naive Bayes (NumPy):
  * Gaussian Naive Bayes
  * Multinomial Naive Bayes
  * Bernoulli Naive Bayes
  * Categorical Naive Bayes
* Softmax Regression (NumPy and PyTorch)
* K-Means Clustering (PyTorch)
* Decision Tree (NumPy):
  * ID3, C4.5, C5.0/See5 Algorithm
  * CART Algorithm
  * CHAID Algorithm
  * CITs Algorithm
  * OC1 Algorithm
  * QUEST, TAO Algorithm
* Random Forest (NumPy)
* Support Vector Machines (PyTorch): linear, rbf, polynomial, sigmoid kernel

### CNN: (PyTorch)
* LeNet
* AlexNet
* NiN
* VGG
* GoogLeNet, Xception, NASNet
* ResNet34, ResNet152, ResNeXt, WideResNet
* DenseNet
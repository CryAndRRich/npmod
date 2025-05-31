# Machine Learning/Deep Learning
This repository consists of **two main parts**: a simple **deep learning framework** ([npmod](https://github.com/CryAndRRich/npmod/tree/main/npmod)) and some basic **machine learning/deep learning models** ([models](https://github.com/CryAndRRich/npmod/tree/main/models)). Since this is for learning purposes, everything is built **from scratch** using NumPy and PyTorch (mostly NumPy)

## npmod: NumPy module
A simple **deep learning framework** built using **pure NumPy**

```
npmod/nn/
├── layers/
│   ├── linear/
│   ├── dropout/
│   ├── flatten/
│   ├── linear/
│   ├── batchnorm/
│   ├── conv/
│   │   ├── conv2d, conv3d/
│   │   └── maxpool2d, maxpool3d/
│   └── rnn/
│       ├── rnn/
│       ├── lstm/
│       └── gru/
├── activations/
│   ├── relu, leaky relu/
│   ├── sigmoid/
│   ├── tanh/
│   └── softmax/
├── container/
|   └── sequential/
├── losses/
|   ├── mae, mse, male/
|   ├── rsquared/
|   ├── mape, wmape/
|   ├── smoothl1/
|   ├── ce, bce/
|   ├── kldiv/
└── optimizer/
    ├── gd, sgd/
    ├── adagrad/
    ├── rmsprop/
    └── adam/
```
## models:
Every model is built using NumPy or PyTorch (or both)

```
models/
├── supervised/
│   ├── classification/
│   │   ├── decision_tree/
│   │   │   ├── id3, c4.5, c5.0 algorithm/
│   │   │   ├── cart algorithm/
│   │   │   ├── chaid algorithm/
│   │   │   ├── cits algorithm/
│   │   │   ├── oc1 algorithm/
│   │   │   └── quest, tao algorithm/
│   │   ├── k_nearest_neighbors/
│   │   ├── logistic_regression/
│   │   ├── naive_bayes/
│   │   │   ├── gaussian_nb/
│   │   │   ├── multinomial_nb/
│   │   │   ├── bernoulli_nb/
│   │   │   └── categorical_nb/
│   │   ├── perceptron_learning/
│   │   ├── softmax_regression/
│   │   └── support_vector_machines/
│   ├── regression/
|   |   ├── decision_tree/
│   │   ├── elastic_net/
│   │   │   ├── lasso_regression/
│   │   │   └── ridge_regression/
│   │   ├── generalized_linear_model/
│   │   ├── huber_regression/
│   │   ├── stepwise_regression/
│   │   ├── theilsen_regression/
│   │   ├── bayes_linear_regression/
│   │   └── linear_regression/
│   └── ensemble/
│       ├── bagging/
│       |   └── random_forest/
|       └── boosting/
│           ├── gradient_boosting/
│           ├── adaboost/
│           ├── lightgbm/
│           ├── xgboost/
│           └── catboost/
├── unsupervised/
│   ├── clustering/
|   |   ├── affinity_propagation/
│   │   ├── agglomerative_clustering/
│   │   ├── dbscan/
│   │   ├── gaussian_mixture_model/
│   │   ├── k_means_clustering/
│   │   ├── optics/
│   │   └── spectral_clustering/
│   └── dimensionality_reduction/
│       ├── isomap/
│       ├── lda/
│       ├── lle/
│       ├── mds/
│       ├── nmf/
│       ├── pca/
│       ├── spectral_embedding/
│       ├── svd/
│       ├── t_sne/
|       └── umap/
└── deep_learning/
    ├── cnn/
    │   ├── lenet/
    │   ├── alexnet/
    │   ├── nin/
    │   ├── vgg/
    │   ├── googlenet, xception, nasnet/
    │   ├── resnet34, resnet152, resnext, wideresnet/
    │   └── densenet/
    ├── rnn/
    │   ├── gru, mgu, sru/
    │   ├── lstm/
    │   ├── indrnn, scrn, ugrnn, yamrnn/
    │   ├── janet/
    │   ├── ran/
    │   └── rhn/
    └── mlp/
```
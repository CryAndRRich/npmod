# Machine Learning/Deep Learning
This repository consists of **two main parts**: a simple **deep learning framework** ([npmod](https://github.com/CryAndRRich/npmod/tree/main/npmod)) and some basic **machine learning/deep learning models** ([models](https://github.com/CryAndRRich/npmod/tree/main/models)). Since this is for learning purposes, everything is built **from scratch** using NumPy and PyTorch (mostly NumPy)

## npmod: NumPy module
A simple **deep learning framework** built using **pure NumPy**

```
npmod/nn/
├── layers/
│   ├── flatten/
│   ├── dropout/
│   ├── droconnect/
│   ├── linear/
│   ├── linear/
│   ├── conv/
│   ├── pooling/
│   ├── batchnorm/
│   ├── groupnorm/
│   ├── weightnorm/
│   │
│   └── rnn/
│       ├── rnn/
│       ├── lstm/
│       └── gru/
│
├── activations/
│   ├── relu/
│   │   ├── relu.py
│   │   ├── leaky_relu.py
│   │   ├── prelu.py
│   │   ├── elu.py
│   │   ├── selu.py
│   │   └── relu6.py
│   │
│   ├── sigmoid/
│   │   ├── sigmoid.py
│   │   ├── hard_sigmoid.py
│   │   └── log_sigmoid.py
│   │
│   ├── tanh/
│   │   ├── tanh.py
│   │   └── hard_tanh.py
│   │
│   ├── softmax/
│   │   ├── softmax.py
│   │   └── log_softmax.py
│   │
│   └── smooth/
│       ├── gelu.py
│       ├── swish.py
│       ├── hard_swish.py
│       ├── softplus.py
│       └── mish.py
│
├── container/
|   └── sequential/
│
├── losses/
|   ├── mae, mse, male/
|   ├── rsquared/
|   ├── mape, wmape/
|   ├── smoothl1/
|   ├── ce, bce/
|   └── kldiv/
│
└── optimizer/
    ├── gd, sgd/
    ├── adagrad/
    ├── rmsprop/
    └── adam/
```
## models:
Every model is built using NumPy or PyTorch

```
models/
├── supervised/
│   ├── classification/
│   │   ├── decision_tree/
│   │   │   ├── id3, c4.5, c5.0/
│   │   │   ├── cart/
│   │   │   ├── chaid/
│   │   │   ├── cits/
│   │   │   ├── oc1/
│   │   │   └── quest, tao/
│   │   │
│   │   ├── k_nearest_neighbors/
│   │   ├── logistic_regression/
│   │   ├── naive_bayes/
│   │   │   ├── gaussian/
│   │   │   ├── multinomial/
│   │   │   ├── bernoulli/
│   │   │   └── categorical/
│   │   │
│   │   ├── perceptron_learning/
│   │   ├── softmax_regression/
│   │   └── support_vector_machines/
│   │
│   ├── regression/
|   |   ├── decision_tree/
│   │   ├── k_nearest_neighbors/
│   │   ├── elastic_net/
│   │   │   ├── lasso_regression/
│   │   │   └── ridge_regression/
│   │   │
│   │   ├── generalized_linear_model/
│   │   ├── huber_regression/
│   │   ├── stepwise_regression/
│   │   ├── theilsen_regression/
│   │   ├── bayes_linear_regression/
│   │   └── linear_regression/
│   │
│   └── ensemble/
│       ├── bagging/
│       |   └── random_forest/
|       └── boosting/
│           ├── gradient_boosting/
│           ├── adaboost/
│           ├── lightgbm/
│           ├── xgboost/
│           └── catboost/
│
├── unsupervised/
│   ├── clustering/
|   |   ├── partition_based/
|   |   |   ├── k_means/
|   |   |   ├── k_medians/
|   |   |   ├── k_medoids/
|   |   |   ├── ik_means/
|   |   |   └── affinity_propagation/
|   |   |
│   │   ├── tree_based/
|   |   |   ├── agglomerative/
|   |   |   ├── divisive/
|   |   |   ├── cure/
|   |   |   └── chameleon/
|   |   |
│   │   ├── density_based/
|   |   |   ├── dbscan/
|   |   |   ├── hdbscan/
|   |   |   ├── denclue/
|   |   |   └── optics/
|   |   |
│   │   ├── grid_based/
|   |   |   ├── gridclus/
|   |   |   ├── sting/
|   |   |   ├── waveclus/
|   |   |   ├── gdilc/
|   |   |   └── amr/
|   |   |
│   │   ├── graph_based/
|   |   |   └── spectral_clustering/
|   |   |
│   │   └── model_based/
|   |       ├── bayesian_gaussian_mixture_model/
|   |       └── gaussian_mixture_model/
|   |  
│   └── dimensionality_reduction/
|       ├── linear/
|       |   ├── pca/
|       |   ├── lda/
|       |   └── nmf/
|       |
│       └── model_based/
|           ├── isomap/
|           ├── lle/
|           ├── mds/
|           ├── se/
|           ├── t_sne/
|           └── umap/
|
└── deep_learning/
    ├── mlp/
    │
    ├── cnn/
    │   ├── lenet/
    │   ├── alexnet/
    │   ├── nin/
    │   ├── vgg/
    │   ├── googlenet/
    │   ├── resnet34, resnet152/
    │   ├── squeezenet/
    │   ├── resnext/
    │   ├── xception/
    │   ├── densenet/
    │   ├── wideresnet/
    │   ├── mobilenet/
    │   ├── nasnet/
    │   ├── shufflenet/
    │   ├── efficientnet/
    │   ├── regnet/
    │   ├── ghostnet/
    │   └── micronet/
    │
    ├── rnn/
    │   ├── gru, mgu, sru/
    │   ├── lstm/
    │   ├── indrnn, scrn, ugrnn, yamrnn/
    │   ├── janet/
    │   ├── ran/
    │   └── rhn/
    │
    ├── gan/
    │   ├── supervised/
    |   |   ├── cgan/
    |   |   ├── pix2pix/
    |   |   └── srgan/
    |   |     
    │   └── supervised/
    |       ├── gan/
    |       ├── dcgan/
    |       ├── lapgan/
    |       ├── wgan/
    |       ├── spade/
    |       └── stylegan/
    │     
    └── autoencoder/
        ├── cae/
        ├── convae/
        ├── vae/
        └── dae/
```
# Machine Learning/Deep Learning
This repository consists of **two main parts**: a simple **deep learning framework** ([npmod](https://github.com/CryAndRRich/npmod/tree/main/npmod)) and some basic **machine learning/deep learning models** ([models](https://github.com/CryAndRRich/npmod/tree/main/models)). Since this is for learning purposes, everything is built **from scratch** using NumPy and PyTorch

All datasets used to test the models can be found via the links provided at the end of this README

## npmod: NumPy module
A simple **deep learning framework** built using **pure NumPy**

```
npmod/nn/
├── layers/
│   ├── flatten
│   ├── dropout
│   ├── droconnect
│   ├── linear
│   ├── linear
│   ├── conv
│   ├── pooling
│   ├── batchnorm
│   ├── groupnorm
│   └── weightnorm
│   
├── rnn/
│   ├── rnn
│   ├── lstm
│   └── gru
│
├── activations/
│   ├── relu/
│   │   ├── relu
│   │   ├── leaky_relu
│   │   ├── prelu
│   │   ├── elu
│   │   ├── selu
│   │   └── relu6
│   │
│   ├── sigmoid/
│   │   ├── sigmoid
│   │   ├── hard_sigmoid
│   │   └── log_sigmoid
│   │
│   ├── tanh/
│   │   ├── tanh
│   │   └── hard_tanh
│   │
│   ├── softmax/
│   │   ├── softmax
│   │   └── log_softmax
│   │
│   └── smooth/
│       ├── gelu
│       ├── swish
│       ├── hard_swish
│       ├── softplus
│       └── mish
│
├── container/
|   └── sequential
│
├── losses/
│   ├── regression/
│   │   ├── mae
│   │   ├── mse
│   │   ├── male
│   │   ├── rsquared
│   │   ├── mape
│   │   ├── wmape
│   │   ├── smoothl1
│   │   ├── huber
│   │   ├── logcosh
│   │   └── quantile
│   │
│   ├── classification/
│   │   ├── ce
│   │   ├── bce
│   │   ├── focalloss
│   │   ├── labelsmoothingce
│   │   └── diceloss
│   │
│   ├── divergence/
│   │   ├── kldiv
│   │   ├── jsdiv
│   │   └── wasserstein
│   │
│   └── ranking/
│       ├── __init__
│       ├── hinge_embedding
│       ├── margin_ranking
│       └── triplet_margin
│
└── optimizer/
    ├── classical/
    │   ├── gd              
    │   └── sgd             
    │
    ├── adaptive/
    │   ├── adagrad
    │   ├── rmsprop
    │   ├── adam
    │   ├── adamw
    │   ├── radam
    │   ├── adabelief
    │   └── lookahead
    │
    └── modern/
        ├── lamb
        ├── novograd
        ├── ranger
        ├── apollo
        ├── sophia
        └── lion
```
## models:
Every model is built using NumPy or PyTorch

```
models/
├── supervised/
│   ├── classification/
│   │   ├── decision_tree/
│   │   │   ├── id3, c4.5, c5.0
│   │   │   ├── cart
│   │   │   ├── chaid
│   │   │   ├── cits
│   │   │   ├── oc1
│   │   │   └── quest, tao
│   │   │
│   │   ├── k_nearest_neighbors
│   │   ├── logistic_regression
│   │   ├── naive_bayes/
│   │   │   ├── gaussian
│   │   │   ├── multinomial
│   │   │   ├── bernoulli
│   │   │   └── categorical
│   │   │
│   │   ├── perceptron_learning
│   │   ├── softmax_regression
│   │   └── support_vector_machines
│   │
│   ├── regression/
|   |   ├── decision_tree
│   │   ├── k_nearest_neighbors
│   │   ├── elastic_net/
│   │   │   ├── lasso_regression
│   │   │   └── ridge_regression
│   │   │
│   │   ├── generalized_linear_model
│   │   ├── huber_regression
│   │   ├── stepwise_regression
│   │   ├── theilsen_regression
│   │   ├── bayes_linear_regression
│   │   └── linear_regression
│   │
│   └── ensemble/
│       ├── bagging/
│       |   └── random_forest
│       |   
|       └── boosting/
│           ├── gradient_boosting
│           ├── adaboost
│           ├── lightgbm
│           ├── xgboost
│           └── catboost
│
├── unsupervised/
│   ├── clustering/
|   |   ├── partition_based/
|   |   |   ├── k_means
|   |   |   ├── k_medians
|   |   |   ├── k_medoids
|   |   |   ├── ik_means
|   |   |   └── affinity_propagation
|   |   |
│   │   ├── tree_based/
|   |   |   ├── agglomerative
|   |   |   ├── divisive
|   |   |   ├── cure
|   |   |   └── chameleon
|   |   |
│   │   ├── density_based/
|   |   |   ├── dbscan
|   |   |   ├── hdbscan
|   |   |   ├── denclue
|   |   |   └── optics
|   |   |
│   │   ├── grid_based/
|   |   |   ├── gridclus
|   |   |   ├── sting
|   |   |   ├── waveclus
|   |   |   ├── gdilc
|   |   |   └── amr
|   |   |
│   │   ├── graph_based/
|   |   |   └── spectral_clustering
|   |   |
│   │   └── model_based/
|   |       ├── bayesian_gaussian_mixture_model
|   |       └── gaussian_mixture_model
|   |  
│   └── dimensionality_reduction/
|       ├── linear/
|       |   ├── pca
|       |   ├── lda
|       |   └── nmf
|       |
│       └── model_based/
|           ├── isomap
|           ├── lle
|           ├── mds
|           ├── se
|           ├── t_sne
|           └── umap
|
└── deep_learning/
    ├── mlp
    │
    ├── cnn/
    │   ├── lenet
    │   ├── alexnet
    │   ├── nin
    │   ├── vgg
    │   ├── googlenet
    │   ├── resnet
    │   ├── squeezenet
    │   ├── resnext
    │   ├── xception
    │   ├── densenet
    │   ├── wideresnet
    │   ├── mobilenet
    │   ├── nasnet
    │   ├── shufflenet
    │   ├── efficientnet
    │   ├── regnet
    │   ├── ghostnet
    │   └── micronet
    │
    ├── rnn/
    │   ├── lstm
    │   ├── gru
    │   ├── mgu
    │   ├── ugrnn
    │   ├── rhn
    │   ├── sru
    │   ├── janet
    │   ├── indrnn
    │   ├── ran
    │   ├── scrn
    │   └── yamrnn
    │
    ├── transformer/
    │   ├── transformer/
    |   |   ├── positional_encoding/
    |   |   |   ├── sinusoidal
    |   |   |   ├── learned
    |   |   |   ├── relative
    |   |   |   ├── rotary
    |   |   |   └── alibias
    |   |   |  
    |   |   ├── normalization/
    |   |   |   ├── layernorm
    |   |   |   ├── rmsnorm
    |   |   |   ├── scalenorm
    |   |   |   └── adanorm  
    |   |   |  
    |   |   ├── attention/
    |   |   |   ├── scaled_dot
    |   |   |   ├── linformer
    |   |   |   ├── performer
    |   |   |   ├── local
    |   |   |   ├── sparse
    |   |   |   └── cosine    
    |   |   |  
    |   |   ├── feed_forward/
    |   |   |   ├── relu
    |   |   |   ├── gelu
    |   |   |   ├── geglu
    |   |   |   ├── swiglu
    |   |   |   ├── glu
    |   |   |   ├── conformer
    |   |   |   └── dropconnect    
    |   |   |  
    |   |   └── learn_rate/
    |   |       ├── noam
    |   |       ├── cosine
    |   |       ├── linear
    |   |       ├── inverse_sqrt
    |   |       ├── polynomial
    |   |       └── constant    
    |   |
    |   └── vit/
    |       ├── vit
    |       ├── deit
    |       └── swin
    |
    ├── gan/
    |   ├── gan
    |   ├── dcgan
    |   ├── lapgan
    |   ├── wgan
    |   ├── progan
    |   ├── biggan
    |   └── stylegan
    │     
    └── autoencoder/
        ├── ae
        ├── regularized_ae
        ├── convolutional_ae
        ├── vae
        ├── aae
        ├── vq_vae
        └── mae
```
## data:
```
data/
├── brisc2025.zip                       # https://arxiv.org/abs/2506.14318
├── cifar10.zip                         # https://www.kaggle.com/datasets/valentynsichkar/cifar10-preprocessed
├── clouds.zip                          # https://www.kaggle.com/datasets/jockeroika/clouds-photos
├── Country-data.csv                    # https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data
├── diabetes.csv                        # https://www.kaggle.com/datasets/abdallamahgoub/diabetes
├── dSprites.zip                        # https://github.com/google-deepmind/dsprites-dataset
├── ffhq.zip                            # https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
├── fitness_dataset.csv                 # https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic
├── pokemon-dataset-10000.zip           # https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000fitness-classification-dataset-synthetic
├── rem.zip                             # https://www.kaggle.com/datasets/andy8744/rezero-rem-anime-faces-for-gan-training
├── satellite.zip                       # https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
├── star_classification.csv             # https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
├── Student_Performance.csv             # https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
├── synthetic_coffee_health_10000.csv   # https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset
├── tom_and_jerry.zip                   # https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification
├── weather5k.zip                       # https://arxiv.org/abs/2406.14399
└── wmt-2014-english-german.zip         # https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
```
# This script tests various classification models, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Importing built-in models from sklearn
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron as PLA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Importing the custom models
from models.supervised.classification.k_nearest_neighbors import KNeighborsClassifier as CustomKNNClassifier
from models.supervised.classification.logistic_regression import LogisticRegression as CustomLogisticRegression
from models.supervised.classification.softmax_regression import SoftmaxRegression as CustomSoftmaxRegression
from models.supervised.classification.naive_bayes import NaiveBayes as CustomNaiveBayes
from models.supervised.classification.perceptron_learning import PerceptronLearning as CustomPLA
from models.supervised.classification.support_vector_machines import SupportVectorMachine as CustomSVC
from models.supervised.classification.decision_tree import DecisionTreeClassifier as CustomDecisionTreeClassifier


# === Model information ===
# 1-4: K-Nearest Neighbors Classifier
# 5-6: Logistic Regression
# 7-8: Softmax Regression
# 9-14: Naive Bayes Classifier
# 15-16: Perceptron Learning Algorithm
# 17-24: Support Vector Machine
# 25-35: Decision Tree Classifier
# ====================


if __name__ == "__main__":
    # === Load all datasets ===
    # Load Iris dataset
    X_iris, y_iris = load_iris(return_X_y=True)
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
        X_iris, y_iris, 
        test_size=0.5, random_state=42
    )

    # Load Diabetes dataset
    # https://www.kaggle.com/datasets/abdallamahgoub/diabetes
    data = pd.read_csv(r"D:\Project\npmod\data\diabetes.csv")
    X_diabetes, y_diabetes = data.iloc[:, :-1], data.iloc[:, -1]
    X_diabetes = scaler.fit_transform(X_diabetes)
    X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(
        X_diabetes, y_diabetes.to_numpy(), 
        test_size=0.2, random_state=42
    )

    # Load Wine dataset
    X_wine, y_wine = load_wine(return_X_y=True)
    X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
        X_wine, y_wine, 
        test_size=0.5, random_state=42
    )

    # Load Digits dataset
    X_digit, y_digit = load_digits(return_X_y=True)
    X_digit = (X_digit > 5).astype(int)  # Binarize 
    X_digit_train, X_digit_test, y_digit_train, y_digit_test = train_test_split(
        X_digit, y_digit, 
        test_size=0.5, random_state=42
    )
    # ====================


    # === KNNClassifier using built-in and custom implementations ===
    model1 = KNNClassifier(n_neighbors=5, weights="uniform")
    model1.fit(X_iris_train, y_iris_train)
    y_pred1 = model1.predict(X_iris_test)

    model2 = CustomKNNClassifier(neighbors=5, weights="uniform")
    model2.fit(X_iris_train, y_iris_train)
    y_pred2 = model2.predict(X_iris_test)

    print("==============================================================")
    print("K-Nearest Neighbors Classifier Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_iris_test, y_pred1))
    print("Accuracy (custom model):", accuracy_score(y_iris_test, y_pred2))

    """
    Accuracy (built-in model): 0.9466666666666667
    Accuracy (custom model): 0.9466666666666667
    """

    model3 = KNNClassifier(n_neighbors=5, weights="distance")
    model3.fit(X_iris_train, y_iris_train)
    y_pred3 = model3.predict(X_iris_test)

    model4 = CustomKNNClassifier(neighbors=5, weights="distance")
    model4.fit(X_iris_train, y_iris_train)
    y_pred4 = model4.predict(X_iris_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_iris_test, y_pred3))
    print("Accuracy (custom model):", accuracy_score(y_iris_test, y_pred4))

    """
    Accuracy (built-in model): 0.96
    Accuracy (custom model): 0.96
    """
    # ====================


    # === Logistic Regression using built-in and custom implementations ===
    model5 = LogisticRegression()
    model5.fit(X_diabetes_train, y_diabetes_train)
    y_pred5 = model5.predict(X_diabetes_test)

    model6 = CustomLogisticRegression(learn_rate=0.1, number_of_epochs=200)
    model6.fit(X_diabetes_train, y_diabetes_train)
    y_pred6 = model6.predict(X_diabetes_test)

    print("==============================================================")
    print("Logistic Regression Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred5))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred6))
    print("==============================================================")
    print("Weights (built-in model):", model5.coef_)
    print("Weights (custom model):", model6.weights)
    print("==============================================================")
    print("Bias (built-in model):", model5.intercept_)
    print("Bias (custom model):", model6.bias)

    """
    Accuracy (built-in model): 0.7532467532467533
    Accuracy (custom model): 0.7532467532467533
    ==============================================================
    Weights (built-in model): [[ 0.21624195  1.06932996 -0.25867641  0.04720329 -0.19899822  0.79237086
      0.22709403  0.43036184]]
    Weights (custom model): [ 0.22062391  0.94353016 -0.18254185  0.01482228 -0.11539018  0.68374544   
      0.22166946  0.39686046]
    ==============================================================
    Bias (built-in model): [-0.85594942]
    Bias (custom model): -0.779529845889643
    ==============================================================
    """
    # ====================


    # === Softmax Regression using built-in and custom implementations ===
    model7 = LogisticRegression(solver='lbfgs')
    model7.fit(X_iris_train, y_iris_train.ravel())
    y_pred7 = model7.predict(X_iris_test)

    model8 = CustomSoftmaxRegression(learn_rate=0.1, number_of_epochs=500, number_of_classes=3)
    model8.fit(X_iris_train, y_iris_train)
    y_pred8 = model8.predict(X_iris_test)

    print("==============================================================")
    print("Softmax Regression Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_iris_test, y_pred7))
    print("Accuracy (custom model):", accuracy_score(y_iris_test, y_pred8))
    print("==============================================================")
    print("Weights (built-in model):", model7.coef_)
    print("Weights (custom model):", model8.weights)
    print("==============================================================")
    print("Bias (built-in model):", model7.intercept_)
    print("Bias (custom model):", model8.bias)

    """
    ==============================================================
    Accuracy (built-in model): 1.0
    Accuracy (custom model): 0.9866666666666667
    ==============================================================
    Weights (built-in model): [[-0.36811645  0.73365587 -2.07229377 -0.88334482]
    [ 0.42583021 -0.37525322 -0.17639551 -0.6114654 ]
    [-0.05771376 -0.35840265  2.24868929  1.49481022]]
    Weights (custom model): [[ 0.88075503  2.42823684 -1.75735347 -0.51584059]
    [ 1.06063745  0.23731058  0.08186056  0.16792777]
    [-0.80971871 -0.85076601  2.48615496  2.7826573 ]]
    ==============================================================
    Bias (built-in model): [  8.21171936   2.15179107 -10.36351043]
    Bias (custom model): [[ 0.31540889  0.52331225 -0.83872114]]
    ==============================================================
    """
    # ====================


    # === Naive Bayes using built-in and custom implementations ===
    model9 = GaussianNB()
    model9.fit(X_iris_train, y_iris_train)
    y_pred9 = model9.predict(X_iris_test)

    model10 = CustomNaiveBayes(distribution='gaussian')
    model10.fit(X_iris_train, y_iris_train)
    y_pred10 = model10.predict(X_iris_test)

    print("==============================================================")
    print("Naive Bayes Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_iris_test, y_pred9))
    print("Accuracy (custom model):", accuracy_score(y_iris_test, y_pred10))

    """
    Accuracy (built-in model): 0.9866666666666667
    Accuracy (custom model): 0.9866666666666667
    """

    model11 = MultinomialNB(alpha=1)
    model11.fit(X_wine_train, y_wine_train)
    y_pred11 = model11.predict(X_wine_test)

    model12 = CustomNaiveBayes(distribution='multinomial', alpha=1)
    model12.fit(X_wine_train, y_wine_train)
    y_pred12 = model12.predict(X_wine_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_wine_test, y_pred11))
    print("Accuracy (custom model):", accuracy_score(y_wine_test, y_pred12))

    """
    Accuracy (built-in model): 0.8876404494382022
    Accuracy (custom model): 0.8876404494382022
    """

    model13 = BernoulliNB(alpha=1)
    model13.fit(X_digit_train, y_digit_train)
    y_pred13 = model13.predict(X_digit_test)

    model14 = CustomNaiveBayes(distribution='bernoulli', alpha=1)
    model14.fit(X_digit_train, y_digit_train)
    y_pred14 = model14.predict(X_digit_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_digit_test, y_pred13))
    print("Accuracy (custom model):", accuracy_score(y_digit_test, y_pred14))

    """
    Accuracy (built-in model): 0.8776418242491657
    Accuracy (custom model): 0.8776418242491657
    """
    # ====================


    # === Perceptron Learning Algorithm using built-in and custom implementations ===
    model15 = PLA()
    model15.fit(X_diabetes_train, y_diabetes_train)
    y_pred15 = model15.predict(X_diabetes_test)

    model16 = CustomPLA(learn_rate=0.1, number_of_epochs=50)
    model16.fit(X_diabetes_train, y_diabetes_train)
    y_pred16 = model16.predict(X_diabetes_test)

    print("==============================================================")
    print("Perceptron Learning Algorithm Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred15))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred16))
    print("==============================================================")
    print("Weights (built-in model):", model15.coef_)
    print("Weights (custom model):", model16.weights)
    print("==============================================================")
    print("Bias (built-in model):", model15.intercept_)
    print("Bias (custom model):", model16.bias)

    """
    ==============================================================
    Accuracy (built-in model): 0.6558441558441559
    Accuracy (custom model): 0.7142857142857143
    ==============================================================
    Weights (built-in model): [[ 1.04982285  0.17788026 -1.48287726 -0.65178376 -0.38727073 -0.05359363
      -0.82561315 -1.12973935]]
    Weights (custom model): [ 0.40748673  0.02389606 -0.29787361 -0.08586876  0.26916244  0.40145637
      0.10207334  0.10813536]
    ==============================================================
    Bias (built-in model): [-3.]
    Bias (custom model): -0.20000000000000004
    ==============================================================
    """
    # ====================


    # === Support Vector Machine using built-in and custom implementations ===
    model17 = SVC(kernel="linear", C=1.0)
    model17.fit(X_diabetes_train, y_diabetes_train)
    y_pred17 = model17.predict(X_diabetes_test)

    model18 = CustomSVC(kernel="linear", C=1.0)
    model18.fit(X_diabetes_train, y_diabetes_train)
    y_pred18 = model18.predict(X_diabetes_test)

    print("==============================================================")
    print("Support Vector Machine Results")
    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred17))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred18))

    """
    Accuracy (built-in model): 0.7597402597402597
    Accuracy (custom model): 0.6883116883116883
    """

    model19 = SVC(kernel="rbf", C=1.0)
    model19.fit(X_diabetes_train, y_diabetes_train)
    y_pred19 = model19.predict(X_diabetes_test)

    model20 = CustomSVC(kernel="rbf", C=1.0)
    model20.fit(X_diabetes_train, y_diabetes_train)
    y_pred20 = model20.predict(X_diabetes_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred19))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred20))

    """
    Accuracy (built-in model): 0.7272727272727273
    Accuracy (custom model): 0.6428571428571429
    """

    model21 = SVC(kernel="poly", C=1.0, degree=3)
    model21.fit(X_diabetes_train, y_diabetes_train)
    y_pred21 = model21.predict(X_diabetes_test)

    model22 = CustomSVC(kernel="poly", C=1.0, degree=3)
    model22.fit(X_diabetes_train, y_diabetes_train)
    y_pred22 = model6.predict(X_diabetes_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred21))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred22))

    """
    Accuracy (built-in model): 0.7467532467532467
    Accuracy (custom model): 0.7532467532467533
    """

    model23 = SVC(kernel="sigmoid", C=1.0)
    model23.fit(X_diabetes_train, y_diabetes_train)
    y_pred23 = model23.predict(X_diabetes_test)

    model24 = CustomSVC(kernel="sigmoid", C=1.0)
    model24.fit(X_diabetes_train, y_diabetes_train)
    y_pred24 = model24.predict(X_diabetes_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_diabetes_test, y_pred23))
    print("Accuracy (custom model):", accuracy_score(y_diabetes_test, y_pred24))

    """
    Accuracy (built-in model): 0.6558441558441559
    Accuracy (custom model): 0.6948051948051948
    """
    # ====================


    # === Decision Tree Classifier using built-in and custom implementations ===
    model25 = DecisionTreeClassifier(criterion='gini')
    model25.fit(X_diabetes_train, y_diabetes_train)
    y_pred25 = model25.predict(X_diabetes_test)

    model26 = DecisionTreeClassifier(criterion='entropy')
    model26.fit(X_diabetes_train, y_diabetes_train)
    y_pred26 = model26.predict(X_diabetes_test)

    model27 = CustomDecisionTreeClassifier(algorithm='ID3')
    model27.fit(X_diabetes_train, y_diabetes_train)
    y_pred27 = model27.predict(X_diabetes_test)

    model28 = CustomDecisionTreeClassifier(algorithm='C4.5')
    model28.fit(X_diabetes_train, y_diabetes_train)
    y_pred28 = model28.predict(X_diabetes_test)

    model29 = CustomDecisionTreeClassifier(algorithm='C5.0')
    model29.fit(X_diabetes_train, y_diabetes_train)
    y_pred29 = model29.predict(X_diabetes_test)

    model30 = CustomDecisionTreeClassifier(algorithm='CART')
    model30.fit(X_diabetes_train, y_diabetes_train)
    y_pred30 = model30.predict(X_diabetes_test)

    model31 = CustomDecisionTreeClassifier(algorithm='CHAID')
    model31.fit(X_diabetes_train, y_diabetes_train)
    y_pred31 = model31.predict(X_diabetes_test)

    model32 = CustomDecisionTreeClassifier(algorithm='CITs')
    model32.fit(X_diabetes_train, y_diabetes_train)
    y_pred32 = model32.predict(X_diabetes_test)

    model33 = CustomDecisionTreeClassifier(algorithm='OCT1')
    model33.fit(X_diabetes_train, y_diabetes_train)
    y_pred33 = model33.predict(X_diabetes_test)

    model34 = CustomDecisionTreeClassifier(algorithm='QUEST')
    model34.fit(X_diabetes_train, y_diabetes_train)
    y_pred34 = model34.predict(X_diabetes_test)

    model35 = CustomDecisionTreeClassifier(algorithm='TAO')
    model35.fit(X_diabetes_train, y_diabetes_train)
    y_pred35 = model35.predict(X_diabetes_test)

    print("==============================================================")
    print("Decision Tree Classifier Results")
    print("==============================================================")
    print("Accuracy (built-in model - gini):", accuracy_score(y_diabetes_test, y_pred25))
    print("Accuracy (built-in model - entropy):", accuracy_score(y_diabetes_test, y_pred26))
    print("Accuracy (custom model - ID3):", accuracy_score(y_diabetes_test, y_pred27))
    print("Accuracy (custom model - C4.5):", accuracy_score(y_diabetes_test, y_pred28))
    print("Accuracy (custom model - C5.0):", accuracy_score(y_diabetes_test, y_pred29))
    print("Accuracy (custom model - CART):", accuracy_score(y_diabetes_test, y_pred30))
    print("Accuracy (custom model - CHAID):", accuracy_score(y_diabetes_test, y_pred31))
    print("Accuracy (custom model - CITs):", accuracy_score(y_diabetes_test, y_pred32))
    print("Accuracy (custom model - OCT1):", accuracy_score(y_diabetes_test, y_pred33))
    print("Accuracy (custom model - QUEST):", accuracy_score(y_diabetes_test, y_pred34))
    print("Accuracy (custom model - TAO):", accuracy_score(y_diabetes_test, y_pred35))

    """
    Accuracy (built-in model - gini): 0.7467532467532467
    Accuracy (built-in model - entropy): 0.7467532467532467
    Accuracy (custom model - ID3): 0.7207792207792207
    Accuracy (custom model - C4.5): 0.7207792207792207
    Accuracy (custom model - C5.0): 0.7142857142857143
    Accuracy (custom model - CART): 0.7337662337662337
    Accuracy (custom model - CHAID): 0.7402597402597403
    Accuracy (custom model - CITs): 0.7467532467532467
    Accuracy (custom model - OCT1): 0.6623376623376623
    Accuracy (custom model - QUEST): 0.7012987012987013
    Accuracy (custom model - TAO): 0.6168831168831169
    """
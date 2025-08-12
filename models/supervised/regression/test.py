# This script tests various classification models, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Importing built-in models from sklearn
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge, ElasticNet, TheilSenRegressor, PoissonRegressor, GammaRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.tree import DecisionTreeRegressor

# Importing the custom model
from linear_regression import LinearRegression as CustomLinearRegression
from linear_regression import PolynomialRegression as CustomPolynomialRegression
from huber_regression import HuberRegression as CustomHuberRegression
from elastic_net import LassoRegression as CustomLassoRegression
from elastic_net import RidgeRegression as CustomRidgeRegression
from elastic_net import ElasticNetRegression as CustomElasticNet
from k_nearest_neighbors import KNeighborsRegressor as CustomKNNRegressor
from decision_tree import DecisionTreeRegressor as CustomDecisionTreeRegressor
from theilsen_regression import TheilSenRegression as CustomTheilSenRegressor
from stepwise_regression import StepwiseForward as CustomStepwiseForward
from stepwise_regression import StepwiseBackward as CustomStepwiseBackward
from bayes_linear_regression import BayesianLinearRegression as CustomBayesianLinearRegression
from generalized_linear_model import GLMRegression as CustomGLMRegression


# === Model information ===
# 1-2: Linear Regression
# 3-4: Huber Regression
# 5-6: Lasso Regression
# 7-8: Ridge Regression
# 9-10: ElasticNet Regression
# 11-12: Polynomial Regression
# 13-16: K-Nearest Neighbors Regression
# 16-18: Decision Tree Regression
# 19-20: TheilSen Regression
# 21-24: Stepwise Regression
# 25-26: Bayesian Linear Regression
# 27-30: Generalized Linear Model Regression
# ====================


# === Load all datasets ===
# Load Student dataset 
# https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
data = pd.read_csv(r"D:\Project\npmod\data\Student_Performance.csv")
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 0, "No": 1})
X_student, y_student = data.iloc[:, :-1], data.iloc[:, -1]
X_student = scaler.fit_transform(X_student)
X_student_train, X_student_test, y_student_train, y_student_test = train_test_split(
    X_student, y_student.to_numpy(), 
    test_size=0.2, random_state=42
)
# ====================


# === Linear Regression using built-in and custom implementations ===
model1 = LinearRegression()
model1.fit(X_student_train, y_student_train)
y_pred1 = model1.predict(X_student_test)

model2 = CustomLinearRegression(learn_rate=0.1, number_of_epochs=100)
model2.fit(X_student_train, y_student_train)
y_pred2 = model2.predict(X_student_test)

print("==============================================================")
print("Linear Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred1))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred2))
print("==============================================================")
print("Weights (built-in model):", model1.coef_)
print("Weights (custom model):", model2.weights)
print("==============================================================")
print("Bias (built-in model):", model1.intercept_)
print("Bias (custom model):", model2.bias)

"""
MSE Loss (built-in model): 4.082628398521852
MSE Loss (custom model): 4.082628397094975
==============================================================
Weights (built-in model): [ 7.38559242 17.63689926 -0.30429188  0.80878696  0.55001995]
Weights (custom model): [ 7.38559242 17.63689926 -0.30429188  0.80878696  0.55001996]  
==============================================================
Bias (built-in model): 55.240756800755946
Bias (custom model): 55.24075678881989
"""
# ====================


# === Huber Regression using built-in and custom implementations ===
model3 = HuberRegressor(epsilon=1.5)
model3.fit(X_student_train, y_student_train)
y_pred3 = model3.predict(X_student_test)
mse3 = mean_squared_error(y_student_test, y_pred3)

model4 = CustomHuberRegression(learn_rate=0.5, number_of_epochs=200, delta=1.5)
model4.fit(X_student_train, y_student_train)
y_pred4 = model4.predict(X_student_test)
mse4 = mean_squared_error(y_student_test, y_pred4)

print("==============================================================")
print("Huber Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred3))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred4))
print("==============================================================")
print("Weights (built-in model):", model3.coef_)
print("Weights (custom model):", model4.weights)
print("==============================================================")
print("Bias (built-in model):", model3.intercept_)
print("Bias (custom model):", model4.bias)

"""
MSE Loss (built-in model): 4.080529667642844
MSE Loss (custom model): 4.081412019622634
==============================================================
Weights (built-in model): [ 7.39013932 17.64737488 -0.31378344  0.80111927  0.54637499]
Weights (custom model): [ 7.39578319 17.64319321 -0.32122526  0.80118163  0.54127465]  
==============================================================
Bias (built-in model): 55.239585023006356
Bias (custom model): 55.236382805440215
"""
# ====================


# === Lasso Regression using built-in and custom implementations ===
model5 = Lasso(alpha=0.1)
model5.fit(X_student_train, y_student_train)
y_pred5 = model5.predict(X_student_test)
mse5 = mean_squared_error(y_student_test, y_pred5)

model6 = CustomLassoRegression(learn_rate=0.1, number_of_epochs=100, reg_rate=0.1)
model6.fit(X_student_train, y_student_train)
y_pred6 = model6.predict(X_student_test)
mse6 = mean_squared_error(y_student_test, y_pred6)

print("==============================================================")
print("Lasso Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred5))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred6))
print("==============================================================")
print("Weights (built-in model):", model5.coef_)
print("Weights (custom model):", model6.weights)
print("==============================================================")
print("Bias (built-in model):", model5.intercept_)
print("Bias (custom model):", model6.bias)

"""
MSE Loss (built-in model): 4.1752724515261725
MSE Loss (custom model): 4.116021261011303
==============================================================
Weights (built-in model): [ 7.2873848  17.5371275  -0.20274874  0.70776313  0.45199991]
Weights (custom model): [ 7.33648858 17.58701341 -0.25352032  0.75827505  0.50100994]
==============================================================
Bias (built-in model): 55.2426377691611
Bias (custom model): 55.24169727275986
"""
# ====================


# === Ridge Regression using built-in and custom implementations ===
model7 = Ridge(alpha=0.1)
model7.fit(X_student_train, y_student_train)
y_pred7 = model7.predict(X_student_test)
mse7 = mean_squared_error(y_student_test, y_pred7)

model8 = CustomRidgeRegression(learn_rate=0.1, number_of_epochs=200, reg_rate=0.1)
model8.fit(X_student_train, y_student_train)
y_pred8 = model8.predict(X_student_test)
mse8 = mean_squared_error(y_student_test, y_pred8)

print("==============================================================")
print("Ridge Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred7))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred8))
print("==============================================================")
print("Weights (built-in model):", model7.coef_)
print("Weights (custom model):", model8.weights)
print("==============================================================")
print("Bias (built-in model):", model7.intercept_)
print("Bias (custom model):", model8.bias)

"""
MSE Loss (built-in model): 4.082686186853378
MSE Loss (custom model): 7.526745667723113
==============================================================
Weights (built-in model): [ 7.38549822 17.63667783 -0.30428879  0.80877745  0.55001677]
Weights (custom model): [ 6.7017591  16.02712936 -0.28128479  0.73918616  0.52437567]
==============================================================
Bias (built-in model): 55.24075763647223
Bias (custom model): 55.246867349590396
"""
# ====================


# === ElasticNet Regression using built-in and custom implementations ===
model9 = ElasticNet(alpha=0.1, l1_ratio=0.5)
model9.fit(X_student_train, y_student_train)
y_pred9 = model9.predict(X_student_test)
mse9 = mean_squared_error(y_student_test, y_pred9)

model10 = CustomElasticNet(learn_rate=0.1, number_of_epochs=100, alpha=0.1, l1_ratio=0.5)
model10.fit(X_student_train, y_student_train)
y_pred10 = model10.predict(X_student_test)
mse10 = mean_squared_error(y_student_test, y_pred10)

print("==============================================================")
print("ElasticNet Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred9))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred10))
print("==============================================================")
print("Weights (built-in model):", model9.coef_)
print("Weights (custom model):", model10.weights)
print("==============================================================")
print("Bias (built-in model):", model9.intercept_)
print("Bias (custom model):", model10.bias)

"""
MSE Loss (built-in model): 5.286318553901164
MSE Loss (custom model): 5.206622891561104
==============================================================
Weights (built-in model): [ 6.98026878 16.74600098 -0.24404229  0.72434883  0.49048275]
Weights (custom model): [ 7.00367157 16.7697588  -0.26820146  0.74839007  0.51384309]
==============================================================
Bias (built-in model): 55.24484539937044
Bias (custom model): 55.244397414452884
"""
# ====================


# === Polynomial Regression using built-in and custom implementations ===
model11 = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("linear", LinearRegression())
])
model11.fit(X_student_train, y_student_train)
y_pred11 = model11.predict(X_student_test)
mse11 = mean_squared_error(y_student_test, y_pred11)

model12 = CustomPolynomialRegression(degree=3, learn_rate=0.1, number_of_epochs=100)
model12.fit(X_student_train, y_student_train)
y_pred12 = model12.predict(X_student_test)
mse12 = mean_squared_error(y_student_test, y_pred12)

print("===============================================================")
print("Polynomial Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred11))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred12))
print("==============================================================")
print("Weights (built-in model):", model11.named_steps["linear"].coef_)
print("Weights (custom model):", model12.weights)
print("==============================================================")
print("Bias (built-in model):", model11.named_steps["linear"].intercept_)
print("Bias (custom model):", model12.bias)

"""
MSE Loss (built-in model): 4.0921512255873695
MSE Loss (custom model): 4.180825272251069
==============================================================
Weights (built-in model): [-2.09284374e-16  3.68587834e+00  8.78610192e+00 -1.20015062e-01
  3.59343974e-01  3.53547700e-01  1.96942313e-02 -1.32223832e-03
  3.03837393e-02  1.52206689e-02  1.26627533e-03 -3.34172580e-02
  1.87733117e-01  1.36994716e-02 -1.61845503e-02  2.49644829e-03
  9.62079536e-03  1.49517565e-02  3.69555174e-02  2.50678792e-02
 -7.70198291e-03 -1.11970931e-02  6.35496450e-03 -4.14757221e-02
 -1.16163129e-02 -5.69546993e-03  5.42481358e-02  4.90634959e-03
  1.09399156e-02  5.21455870e-02  3.68524633e+00  2.65416482e-02
 -1.18531496e-03 -1.43581923e-02 -4.07368793e-02 -5.86450122e-03
  3.24370464e-02 -1.17975383e-03  5.39725977e-02  2.13203195e-02
  8.78219686e+00  1.54657244e-02 -6.36502185e-03  2.16486557e-02
 -2.33768539e-02 -1.62462275e-02 -1.20066990e-01  3.59143851e-01
  3.53236687e-01  1.26759525e-03  1.21264162e-02 -2.22961254e-02
  2.84147134e-02 -3.44315364e-02  5.14345526e-04 -7.69564564e-02]
Weights (custom model): [ 3.51645008e+00  8.43676367e+00  4.47288176e-01  3.63583256e-01
  3.56866748e-01  9.69404353e-02  3.86268281e-03  2.34421193e-02
  1.23429195e-02 -4.63013361e-03  4.86117625e-02  1.70394729e-01
  1.33632177e-02 -1.33774838e-02  2.74195511e+01  1.04254539e-02
  1.37288931e-02  1.25287634e-01  2.69585382e-02  7.31996725e-02
  9.52578628e-02  9.71900789e-02 -3.55139817e-02 -1.44243036e-02
 -1.25365506e-02  9.86282145e-02  5.93976114e-03  1.42911077e-02
  4.83346060e-02  3.51596246e+00  2.72481209e-02 -5.21351278e-04
  3.11504724e-02 -3.88442401e-02  3.28504534e-02  2.44369486e-01
 -5.01889310e-03  4.96069381e-02  2.11275976e-02  8.43321927e+00
  1.22368627e-02 -1.14489804e-02  1.14424726e-01 -2.34798260e-02
  6.79612792e-02 -1.23069333e-01  3.63366395e-01  3.56581172e-01
 -1.03637232e-04  1.70694327e-02 -1.94466610e-02  2.68108955e-02
 -3.75170829e-02  1.17623319e-03 -7.39371143e-02]
==============================================================
Bias (built-in model): 55.21733972647421
Bias (custom model): 27.42885524043666
"""
# ====================


# === K-Nearest Neighbors Regression using built-in and custom implementations ===
model13 = KNNRegressor(n_neighbors=10, weights="uniform")
model13.fit(X_student_train, y_student_train)
y_pred13 = model13.predict(X_student_test)

model14 = CustomKNNRegressor(neighbors=10, weights="uniform")
model14.fit(X_student_train, y_student_train)
y_pred14 = model14.predict(X_student_test)

print("==============================================================")
print("K-Nearest Neighbors Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred13))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred14))

"""
MSE Loss (built-in model): 8.123735
MSE Loss (custom model): 8.12057
"""

model15 = KNNRegressor(n_neighbors=10, weights="distance")
model15.fit(X_student_train, y_student_train)
y_pred15 = model15.predict(X_student_test)

model16 = CustomKNNRegressor(neighbors=10, weights="distance")
model16.fit(X_student_train, y_student_train)
y_pred16 = model16.predict(X_student_test)

print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred15))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred16))

"""
MSE Loss (built-in model): 7.3514669993167345
MSE Loss (custom model): 7.348884951762426
"""
# ====================


# === Decision Tree Regression using built-in and custom implementations ===
model17 = DecisionTreeRegressor(max_depth=5, random_state=42)
model17.fit(X_student_train, y_student_train)
y_pred17 = model17.predict(X_student_test)

model18 = CustomDecisionTreeRegressor(max_depth=5)
model18.fit(X_student_train, y_student_train)
y_pred18 = model18.predict(X_student_test)

print("===============================================================")
print("Decision Tree Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred17))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred18))

"""
MSE Loss (built-in model): 12.987047842797772
MSE Loss (custom model): 12.987047842797775
"""
# ====================


# === TheilSen Regression using built-in and custom implementations ===
model19 = TheilSenRegressor(random_state=42)
model19.fit(X_student_train, y_student_train)
y_pred19 = model19.predict(X_student_test)

model20 = CustomTheilSenRegressor()
model20.fit(X_student_train, y_student_train)
y_pred20 = model20.predict(X_student_test)

print("==============================================================")
print("TheilSen Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred19))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred20))
print("==============================================================")
print("Weights (built-in model):", model19.coef_)
print("Weights (custom model):", model20.slope)
print("==============================================================")
print("Bias (built-in model):", model19.intercept_)
print("Bias (custom model):", model20.intercept)

"""
MSE Loss (built-in model): 4.078595043301543
MSE Loss (custom model): 291.73596942373035
==============================================================
Weights (built-in model): [ 7.38596444 17.65314682 -0.28479828  0.86405828  0.58933865]
Weights (custom model): [0.59005197 2.15006428 0.         0.         0.        ]
==============================================================
Bias (built-in model): 55.087810309870605
Bias (custom model): 55.47746733236512
"""
# ====================


# === Stepwise Regression using custom implementations ===
estimator21 = LinearRegression()
model21 = SequentialFeatureSelector(estimator21, n_features_to_select='auto', direction='forward')
model21.fit(X_student_train, y_student_train)
X_student_train_forward = model21.transform(X_student_train)
estimator21.fit(X_student_train_forward, y_student_train)
X_student_test_forward = model21.transform(X_student_test)
y_pred21 = estimator21.predict(X_student_test_forward)

model22 = CustomStepwiseForward(learn_rate=0.1, number_of_epochs=100)
model22.fit(X_student_train_forward, y_student_train, CustomLinearRegression)
y_pred22 = model22.predict(X_student_test)

print("==============================================================")
print("Stepwise Forward Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred21))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred22))
print("==============================================================")
print("Weights (built-in model):", estimator21.coef_)
print("Weights (custom model):", model22.model.weights)
print("==============================================================")
print("Bias (built-in model):", estimator21.intercept_)
print("Bias (custom model):", model22.model.bias)

"""
MSE Loss (built-in model): 5.241921186551513
MSE Loss (custom model): 5.241921184318979
==============================================================
Weights (built-in model): [ 7.40027476 17.6435474 ]
Weights (custom model): [17.64354739  7.40027476]
==============================================================
Bias (built-in model): 55.25229939874983
Bias (custom model): 55.252299387667314
"""

estimator23 = LinearRegression()
model23 = SequentialFeatureSelector(estimator23, n_features_to_select='auto', direction='backward')
model23.fit(X_student_train, y_student_train)
X_student_train_backward = model23.transform(X_student_train)
estimator23.fit(X_student_train_backward, y_student_train)
X_student_test_backward = model23.transform(X_student_test)
y_pred23 = estimator23.predict(X_student_test_backward)

model24 = CustomStepwiseBackward(learn_rate=0.1, number_of_epochs=100)
model24.fit(X_student_train_backward, y_student_train, CustomLinearRegression)
y_pred24 = model24.predict(X_student_test)

print("==============================================================")
print("Stepwise Backward Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred23))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred24))
print("==============================================================")
print("Weights (built-in model):", estimator23.coef_)
print("Weights (custom model):", model24.model.weights)
print("==============================================================")
print("Bias (built-in model):", estimator23.intercept_)
print("Bias (custom model):", model24.model.bias)

"""
MSE Loss (built-in model): 4.545107899420578
MSE Loss (custom model): 6.406344203490449
==============================================================
Weights (built-in model): [ 7.39448275 17.64368165  0.80276678]
Weights (custom model): [ 7.39448275 17.64368165  0.80276678]
==============================================================
Bias (built-in model): 55.24705801127942
Bias (custom model): 55.247058000048106
"""
# ====================


# === Bayesian Linear Regression using built-in and custom implementations ===
model25 = BayesianRidge()
model25.fit(X_student_train, y_student_train)
y_pred25 = model25.predict(X_student_test)

model26 = CustomBayesianLinearRegression(sigma2=0.1, tau2=0.1)
model26.fit(X_student_train, y_student_train)
y_pred26 = model26.predict(X_student_test)

print("==============================================================")
print("Bayesian Linear Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred25))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred26))

"""
MSE Loss (built-in model): 4.082661264842157
MSE Loss (custom model): 4.082141023479416
"""
# ====================


# === Generalized Linear Model Regression using built-in and custom implementations ===
model27 = PoissonRegressor()
model27.fit(X_student_train, y_student_train)
y_pred27 = model27.predict(X_student_test)

model28 = CustomGLMRegression(distribution="poisson", max_iter=500, tol=1e-4)
model28.fit(X_student_train, y_student_train)
y_pred28 = model28.predict(X_student_test)

print("==============================================================")
print("Generalized Linear Model Regression Results")
print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred27))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred28))

"""
MSE Loss (built-in model): 15.75263478185679
MSE Loss (custom model): 719.5589283292254
"""

model29 = GammaRegressor()
model29.fit(X_student_train, y_student_train)
y_pred29 = model29.predict(X_student_test)

model30 = CustomGLMRegression(distribution="gamma", max_iter=500, tol=1e-4)
model30.fit(X_student_train, y_student_train)
y_pred30 = model30.predict(X_student_test)

print("==============================================================")
print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred29))
print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred30))

"""
MSE Loss (built-in model): 98.39994880451428
MSE Loss (custom model): 3522.646741194097
"""
# ====================
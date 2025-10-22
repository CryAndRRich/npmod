# This script tests various boosting models, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Importing built-in models
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from xgboost import XGBRegressor as XGR
from xgboost import XGBClassifier as XGC
from catboost import CatBoostRegressor as CBR
from catboost import CatBoostClassifier as CBC
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import AdaBoostClassifier as ABC
from lightgbm import LGBMRegressor as LGBMR
from lightgbm import LGBMClassifier as LGBMC

# Importing the custom models
from models.supervised.ensemble.boosting.gradient_boosting.gbm_regressor import GradientBoostingRegressor as CustomGBR
from models.supervised.ensemble.boosting.gradient_boosting.gbm_classifier import GradientBoostingClassifier as CustomGBC
from models.supervised.ensemble.boosting.xgboost.xgb_regressor import XGBRegressor as CustomXGR
from models.supervised.ensemble.boosting.xgboost.xgb_classifier import XGBClassifier as CustomXGC
from models.supervised.ensemble.boosting.catboost.cat_regressor import CatBoostRegressor as CustomCBR
from models.supervised.ensemble.boosting.catboost.cat_classifier import CatBoostClassifier as CustomCBC
from models.supervised.ensemble.boosting.adaboost.ada_regressor import AdaBoostRegressor as CustomABR
from models.supervised.ensemble.boosting.adaboost.ada_classifier import AdaBoostClassifier as CustomABC
from models.supervised.ensemble.boosting.lightgbm.lgbm_regressor import LightGBMRegressor as CustomLGBMR
from models.supervised.ensemble.boosting.lightgbm.lgbm_classifier import LightGBMClassifier as CustomLGBMC


# === Model information ===
# 1-4: Gradient Boosting
# 5-8: XGBoost
# 9-12: CatBoost
# 13-16: AdaBoost
# 17-20: LightGBM
# ====================


if __name__ == "__main__":
    # === Load all datasets ===
    # Load student dataset
    # https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression
    data_student = pd.read_csv(r"D:\Project\npmod\data\Student_Performance.csv")
    data_student["Extracurricular Activities"] = data_student["Extracurricular Activities"].map({"Yes": 0, "No": 1})
    X_student, y_student = data_student.iloc[:, :-1], data_student.iloc[:, -1]
    X_student = scaler.fit_transform(X_student)
    y_student.to_numpy().ravel()
    X_student_train, X_student_test, y_student_train, y_student_test = train_test_split(
        X_student, y_student.to_numpy(), 
        test_size=0.2, random_state=42
    )

    # Load stellar dataset
    # https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
    data_stellar = pd.read_csv(r"D:\Project\npmod\data\star_classification.csv")
    labels = data_stellar['class'].map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})
    data_stellar.drop(columns=['obj_ID', 'run_ID', 'field_ID', 'spec_obj_ID', 'rerun_ID', 'fiber_ID', 'class'], inplace=True)
    data_stellar['class'] = labels

    X_stellar, y_stellar = data_stellar.iloc[:1000, :-1], data_stellar.iloc[:1000, -1]
    X_stellar = scaler.fit_transform(X_stellar)
    X_stellar_train, X_stellar_test, y_stellar_train, y_stellar_test = train_test_split(
        X_stellar, y_stellar.to_numpy(), 
        test_size=0.2, random_state=42
    )
    # ====================


    # === Gradient Boosting using built-in and custom implementations ===
    model1 = GBR(learning_rate=0.1, n_estimators=200, max_depth=3)
    model1.fit(X_student_train, y_student_train)
    y_pred1 = model1.predict(X_student_test)

    model2 = CustomGBR(learn_rate=0.1, number_of_epochs=200, max_depth=3)
    model2.fit(X_student_train, y_student_train)
    y_pred2 = model2.predict(X_student_test)

    print("==============================================================")
    print("Gradient Boosting Results")
    print("==============================================================")
    print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred1))
    print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred2))

    """
    MSE Loss (built-in model): 4.2946015241184305
    MSE Loss (custom model): 4.294601524118429
    """

    model3 = GBC(learning_rate=0.1, n_estimators=10, max_depth=3)
    model3.fit(X_stellar_train, y_stellar_train)
    y_pred3 = model3.predict(X_stellar_test)

    model4 = CustomGBC(learn_rate=0.1, number_of_epochs=10, max_depth=3)
    model4.fit(X_stellar_train, y_stellar_train)
    y_pred4 = model4.predict(X_stellar_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_stellar_test, y_pred3))
    print("Accuracy (custom model):", accuracy_score(y_stellar_test, y_pred4))

    """
    Accuracy (built-in model): 0.975
    Accuracy (custom model): 0.965  
    """
    # ====================


    # === XGBoost using built-in and custom implementations ===
    model5 = XGR(learning_rate=0.1, n_estimators=200, max_depth=3)
    model5.fit(X_student_train, y_student_train)
    y_pred5 = model5.predict(X_student_test)

    model6 = CustomXGR(learn_rate=0.1, n_estimators=200, max_depth=3)
    model6.fit(X_student_train, y_student_train)
    y_pred6 = model6.predict(X_student_test)

    print("==============================================================")
    print("XGBoost Results")
    print("==============================================================")
    print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred5))
    print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred6))

    """
    MSE Loss (built-in model): 4.308990181821384
    MSE Loss (custom model): 4.306308872227532
    """

    model7 = XGC(learning_rate=0.1, n_estimators=10, max_depth=3)
    model7.fit(X_stellar_train, y_stellar_train)
    y_pred7 = model7.predict(X_stellar_test)

    model8 = CustomXGC(n_classes=3, learn_rate=0.1, n_estimators=10, max_depth=3)
    model8.fit(X_stellar_train, y_stellar_train)
    y_pred8 = model8.predict(X_stellar_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_stellar_test, y_pred7))
    print("Accuracy (custom model):", accuracy_score(y_stellar_test, y_pred8))

    """
    Accuracy (built-in model): 0.98
    Accuracy (custom model): 0.975
    """
    # ====================


    # === CatBoost using built-in and custom implementations ===
    model9 = CBR(learning_rate=0.1, iterations=200, max_depth=3, verbose=0)
    model9.fit(X_student_train, y_student_train)
    y_pred9 = model9.predict(X_student_test)

    model10 = CustomCBR(learn_rate=0.1, n_estimators=200, max_depth=3)
    model10.fit(X_student_train, y_student_train)
    y_pred10 = model10.predict(X_student_test)

    print("==============================================================")
    print("CatBoost Results")
    print("==============================================================")
    print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred9))
    print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred10))
    
    """
    MSE Loss (built-in model): 4.310222738427117
    MSE Loss (custom model): 4.30035972642994
    """

    model11 = CBC(learning_rate=0.1, iterations=10, max_depth=3, verbose=0)
    model11.fit(X_stellar_train, y_stellar_train)
    y_pred11 = model11.predict(X_stellar_test)

    model12 = CustomCBC(learn_rate=0.1, n_estimators=10, max_depth=3)
    model12.fit(X_stellar_train, y_stellar_train)
    y_pred12 = model12.predict(X_stellar_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_stellar_test, y_pred11))
    print("Accuracy (custom model):", accuracy_score(y_stellar_test, y_pred12))

    """
    Accuracy (built-in model): 0.97
    Accuracy (custom model): 0.975
    """
    # ====================


    # === AdaBoost using built-in and custom implementations ===
    model13 = ABR(
        estimator=DecisionTreeRegressor(max_depth=3),
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )
    model13.fit(X_student_train, y_student_train)
    y_pred13 = model13.predict(X_student_test)

    model14 = CustomABR(learn_rate=0.1, number_of_epochs=200, max_depth=3)
    model14.fit(X_student_train, y_student_train)
    y_pred14 = model14.predict(X_student_test)

    print("==============================================================")
    print("AdaBoost Results")
    print("==============================================================")
    print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred13))
    print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred14))
    
    """
    MSE Loss (built-in model): 11.396989083702142
    MSE Loss (custom model): 7.4792731447995005
    """

    model15 = ABC(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=10,
        learning_rate=0.1,
        random_state=42
    )
    model15.fit(X_stellar_train, y_stellar_train)
    y_pred15 = model15.predict(X_stellar_test)

    model16 = CustomABC(learn_rate=0.1, number_of_epochs=10, max_depth=3)
    model16.fit(X_stellar_train, y_stellar_train)
    y_pred16 = model16.predict(X_stellar_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_stellar_test, y_pred15))
    print("Accuracy (custom model):", accuracy_score(y_stellar_test, y_pred16))

    """
    Accuracy (built-in model): 0.98
    Accuracy (custom model): 0.97
    """
    # ====================


    # === LightGBM using built-in and custom implementations ===
    model17 = LGBMR(learning_rate=0.1, n_estimators=200)
    model17.fit(X_student_train, y_student_train)
    y_pred17 = model17.predict(X_student_test)

    model18 = CustomLGBMR(learn_rate=0.1, n_estimators=200)
    model18.fit(X_student_train, y_student_train)
    y_pred18 = model18.predict(X_student_test)

    print("==============================================================")
    print("LightGBM Results")
    print("==============================================================")
    print("MSE Loss (built-in model):", mean_squared_error(y_student_test, y_pred17))
    print("MSE Loss (custom model):", mean_squared_error(y_student_test, y_pred18))
    
    """
    MSE Loss (built-in model): 4.399633636511799
    MSE Loss (custom model): 4.395947415646208
    """

    model19 = LGBMC(learning_rate=0.1, n_estimators=10)
    model19.fit(X_stellar_train, y_stellar_train)
    y_pred19 = model19.predict(X_stellar_test)

    model20 = CustomLGBMC(learn_rate=0.1, n_estimators=10)
    model20.fit(X_stellar_train, y_stellar_train)
    y_pred20 = model20.predict(X_stellar_test)

    print("==============================================================")
    print("Accuracy (built-in model):", accuracy_score(y_stellar_test, y_pred19))
    print("Accuracy (custom model):", accuracy_score(y_stellar_test, y_pred20))

    """
    Accuracy (built-in model): 0.975
    Accuracy (custom model): 0.98
    """
    # ====================

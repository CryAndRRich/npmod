# This script tests various mutilayer perceptron classifier and regressor models, both built-in and custom implementations
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Importing built-in models from sklearn
from sklearn.neural_network import MLPClassifier as SKMLPC
from sklearn.neural_network import MLPRegressor as SKMLPR

# Importing the custom models
from models.deep_learning.multilayer_perceptron import MLPClassifier as CustomMLPC
from models.deep_learning.multilayer_perceptron import MLPRegressor as CustomMLPR


# === Model information ===
# 1-3: MLP Classifier
# 4-6: MLP Regressor
# ====================


# Function to initialize weights
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # === Load all datasets ===
    # Load digits dataset
    X_digit, y_digit = load_digits(return_X_y=True)
    X_digit = StandardScaler().fit_transform(X_digit)
    X_digit_train_numpy, X_digit_test_numpy, y_digit_train_numpy, y_digit_test_numpy = train_test_split(
        X_digit, y_digit, 
        test_size=0.2, random_state=42
    )

    X_digit_train_torch = torch.tensor(X_digit_train_numpy, dtype=torch.float32)
    y_digit_train_torch = torch.tensor(y_digit_train_numpy, dtype=torch.long)
    X_digit_test_torch = torch.tensor(X_digit_test_numpy, dtype=torch.float32)
    y_digit_test_torch = torch.tensor(y_digit_test_numpy, dtype=torch.long)

    # Load student dataset
    data = pd.read_csv(r"D:\Project\npmod\data\Student_Performance.csv")
    data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 0, "No": 1})
    X_student, y_student = data.iloc[:, :-1], data.iloc[:, -1]
    X_student = StandardScaler().fit_transform(X_student)
    X_student_train_numpy, X_student_test_numpy, y_student_train_numpy, y_student_test_numpy = train_test_split(
        X_student, y_student.to_numpy(), 
        test_size=0.2, random_state=42
    )

    X_student_train_torch = torch.tensor(X_student_train_numpy, dtype=torch.float32)
    y_student_train_torch = torch.tensor(y_student_train_numpy, dtype=torch.float32)
    X_student_test_torch = torch.tensor(X_student_test_numpy, dtype=torch.float32)
    y_student_test_torch = torch.tensor(y_student_test_numpy, dtype=torch.float32)
    # ====================


    # === MLPClassifier using built-in and custom implementations ===
    model1 = SKMLPC(hidden_layer_sizes=(128, 64), 
                    activation="relu", 
                    solver="sgd",
                    learning_rate_init=0.01,
                    max_iter=200,
                    batch_size=32,
                    random_state=42)
    
    model1.fit(X_digit_train_numpy, y_digit_train_numpy)
    y_pred1 = model1.predict(X_digit_test_numpy)

    model2 = nn.Sequential(
        nn.Linear(64, 128), 
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)     
    )
    model2.apply(init_weights)
    criterion2 = nn.CrossEntropyLoss()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

    for epoch in range(200):
        permutation = torch.randperm(X_digit_train_torch.size(0))
        for i in range(0, X_digit_train_torch.size(0), 32):
            indices = permutation[i:i+32]
            X_batch, y_batch = X_digit_train_torch[indices], y_digit_train_torch[indices]
            
            optimizer2.zero_grad()
            output = model2(X_batch)
            loss = criterion2(output, y_batch)
            loss.backward()
            optimizer2.step()

    with torch.no_grad():
        y_pred2 = model2(X_digit_test_torch).argmax(dim=1)

    model3 = CustomMLPC(input_dim=64, 
                        hidden_layers=[128, 64], 
                        output_dim=10, 
                        lr=0.01, 
                        epochs=200, 
                        batch_size=32)
    
    model3.fit(X_digit_train_numpy, y_digit_train_numpy)
    y_pred3 = model3.predict(X_digit_test_numpy)

    print("==============================================================")
    print("MLP Classifier Results")
    print("==============================================================")
    print("Accuracy (sklearn):", accuracy_score(y_digit_test_numpy, y_pred1))
    print("Accuracy (pytorch):", accuracy_score(y_digit_test_numpy, y_pred2.numpy()))
    print("Accuracy (custom):", accuracy_score(y_digit_test_numpy, y_pred3))

    """
    Accuracy (sklearn): 0.975
    Accuracy (pytorch): 0.9694444444444444
    Accuracy (custom): 0.9722222222222222
    """
    # ====================


    # === MLPRegressor using built-in and custom implementations ===
    model4 = SKMLPR(hidden_layer_sizes=(8, 4), 
                    activation="relu", 
                    solver="sgd",
                    learning_rate_init=0.001,
                    max_iter=200,
                    batch_size=32,
                    random_state=42)
    
    model4.fit(X_student_train_numpy, y_student_train_numpy)
    y_pred4 = model4.predict(X_student_test_numpy)

    model5 = nn.Sequential(
        nn.Linear(5, 8), 
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)    
    )
    model5.apply(init_weights)
    criterion5 = nn.MSELoss()
    optimizer5 = optim.SGD(model5.parameters(), lr=0.001)

    for epoch in range(200):
        permutation = torch.randperm(X_student_train_torch.size(0))
        for i in range(0, X_student_train_torch.size(0), 32):
            indices = permutation[i:i+32]
            X_batch, y_batch = X_student_train_torch[indices], y_student_train_torch[indices]
            y_batch = y_batch.view(-1, 1)
            optimizer5.zero_grad()
            output = model5(X_batch)
            loss = criterion5(output, y_batch)
            loss.backward()
            optimizer5.step()
    
    with torch.no_grad():
        y_pred5 = model5(X_student_test_torch).squeeze()
    
    model6 = CustomMLPR(input_dim=5, 
                        hidden_layers=[8, 4], 
                        output_dim=1, 
                        lr=0.001, 
                        epochs=200, 
                        batch_size=32)
    
    model6.fit(X_student_train_numpy, y_student_train_numpy)
    y_pred6 = model6.predict(X_student_test_numpy)

    print("==============================================================")
    print("MLP Regressor Results")
    print("==============================================================")
    print("MSE (sklearn):", mean_squared_error(y_student_test_numpy, y_pred4))
    print("MSE (pytorch):", mean_squared_error(y_student_test_numpy, y_pred5.numpy()))
    print("MSE (custom):", mean_squared_error(y_student_test_numpy, y_pred6))
    
    """
    MSE (sklearn): 4.540991313358924
    MSE (pytorch): 4.6874165603788205
    MSE (custom): 4.22615620995928
    """
    # ====================

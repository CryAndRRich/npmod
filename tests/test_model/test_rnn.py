# This script tests various RNN models
# The results are under print function calls in case you dont want to run the code

import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

# Importing the custom models
from models.deep_learning.rnn.LSTM import LSTM
from models.deep_learning.rnn.GRU import GRU
from models.deep_learning.rnn.MGU import MGU
from models.deep_learning.rnn.UGRNN import UGRNN
from models.deep_learning.rnn.RHN import RHN
from models.deep_learning.rnn.SRU import SRU
from models.deep_learning.rnn.JANET import JANET
from models.deep_learning.rnn.IndRNN import IndRNN
from models.deep_learning.rnn.RAN import RAN
from models.deep_learning.rnn.SCRN import SCRN
from models.deep_learning.rnn.YamRNN import YamRNN

# Define Dataset class for weather data
class WeatherSequenceDataset(Dataset):
    def __init__(self, 
                 X: np.ndarray, 
                 lookback: int, 
                 horizon: int) -> None:
        """
        Dataset for weather time series data
        
        Parameters:
            X: Numpy array of shape (N, T, V) where N is number of stations, T is number of time steps, and V is number of features
            lookback: Number of past time steps to use as input
            horizon: Number of future time steps to predict
        """
        self.X = X.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.N, self.T, self.V = X.shape

        self.samples_per_station = self.T - lookback - horizon + 1
        self.total_samples = self.N * self.samples_per_station

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        station_idx = idx // self.samples_per_station
        start_idx = idx % self.samples_per_station

        data = self.X[station_idx]
        X_seq = data[start_idx : start_idx + self.lookback]
        y_seq = data[start_idx + self.lookback : start_idx + self.lookback + self.horizon]
        return torch.from_numpy(X_seq), torch.from_numpy(y_seq)

if __name__ == "__main__":
    # === Load Dataset === 
    # Load Weather5k dataset
    weather_dir = r"D:\Project\npmod\data\weather5k\WEATHER-5K\global_weather_stations"

    feature_cols = ["TMP", "DEW", "WND_ANGLE", "WND_RATE", "SLP"]

    station_files = sorted(os.listdir(weather_dir))[:5]  # Limit to first 5 stations for testing

    data_list = []
    for fname in tqdm(station_files, desc="Read data files"):
        fpath = os.path.join(weather_dir, fname)
        df = pd.read_csv(fpath, parse_dates=["DATE"])
        df = df.sort_values("DATE")
        
        arr = df[feature_cols].to_numpy(dtype=np.float32)
        data_list.append(arr)

    T = data_list[0].shape[0]
    V = len(feature_cols)
    X = np.stack(data_list, axis=0) 

    train_ratio, test_ratio = 0.8, 0.2
    T_total = X.shape[1]
    train_end = int(T_total * train_ratio)

    X_weather_train = X[:, :train_end, :]
    X_weather_test = X[:, train_end:, :]
    
    # Data Normalization
    mu = X_weather_train.reshape(-1, V).mean(axis=0)
    sigma = X_weather_train.reshape(-1, V).std(axis=0)

    X_weather_train = (X_weather_train - mu) / sigma
    X_weather_test = (X_weather_test - mu) / sigma

    lookback = 48     
    forecast_horizon = 24
    batch_size = 1024
    input_size = output_size = X_weather_train.shape[-1]

    train_weather_dataset = WeatherSequenceDataset(X_weather_train, lookback, forecast_horizon)
    test_weather_dataset = WeatherSequenceDataset(X_weather_test, lookback, forecast_horizon)

    train_weather_loader = DataLoader(train_weather_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_weather_loader = DataLoader(test_weather_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    y_weather_test = []
    for _, targets in test_weather_loader:
        y_weather_test.append(targets.numpy())

    y_weather_test = np.concatenate(y_weather_test, axis=0)  
    # ====================


    # === Test RNN === 
    models = {
        "LSTM": LSTM(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "GRU": GRU(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "MGU": MGU(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "UGRNN": UGRNN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "RHN": RHN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "SRU": SRU(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "JANET": JANET(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "IndRNN": IndRNN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "RAN": RAN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "SCRN": SCRN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
        "YamRNN": YamRNN(learn_rate=2e-4, number_of_epochs=5, input_size=input_size, output_size=output_size, forecast_horizon=forecast_horizon),
    }

    for name, model in models.items():
        print("==============================================================")
        print(f"{name} Result")
        print("==============================================================")

        model.fit(train_loader=train_weather_loader, verbose=True)

        preds = model.predict(test_loader=test_weather_loader)
        mse = mean_squared_error(y_weather_test.reshape(-1), preds.reshape(-1))
        mae = mean_absolute_error(y_weather_test.reshape(-1), preds.reshape(-1))
        print(f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f}")

    """
    ==============================================================
    LSTM Result
    ==============================================================
    Epoch [1/5], Loss: 0.5537
    Epoch [2/5], Loss: 0.4864
    Epoch [3/5], Loss: 0.4729
    Epoch [4/5], Loss: 0.4667
    Epoch [5/5], Loss: 0.4623
    Test MSE: 0.4802 | Test MAE: 0.5022

    ==============================================================
    GRU Result
    ==============================================================
    Epoch [1/5], Loss: 0.5185
    Epoch [2/5], Loss: 0.4795
    Epoch [3/5], Loss: 0.4695
    Epoch [4/5], Loss: 0.4635
    Epoch [5/5], Loss: 0.4594
    Test MSE: 0.4787 | Test MAE: 0.4986
    
    ==============================================================
    MGU Result
    ==============================================================
    Epoch [1/5], Loss: 0.5161
    Epoch [2/5], Loss: 0.4789
    Epoch [3/5], Loss: 0.4702
    Epoch [4/5], Loss: 0.4653
    Epoch [5/5], Loss: 0.4615
    Test MSE: 0.4819 | Test MAE: 0.5006

    ==============================================================
    UGRNN Result
    ==============================================================
    Epoch [1/5], Loss: 0.5244
    Epoch [2/5], Loss: 0.4862
    Epoch [3/5], Loss: 0.4774
    Epoch [4/5], Loss: 0.4722
    Epoch [5/5], Loss: 0.4686
    Test MSE: 0.4797 | Test MAE: 0.5000

    ==============================================================
    RHN Result
    ==============================================================
    Epoch [1/5], Loss: 57.3079
    Epoch [2/5], Loss: 5.4795
    Epoch [3/5], Loss: 2.7872
    Epoch [4/5], Loss: 1.8501
    Epoch [5/5], Loss: 1.4074
    Test MSE: 0.8722 | Test MAE: 0.7018
    
    ==============================================================
    SRU Result
    ==============================================================
    Epoch [1/5], Loss: 0.5345
    Epoch [2/5], Loss: 0.4979
    Epoch [3/5], Loss: 0.4922
    Epoch [4/5], Loss: 0.4872
    Epoch [5/5], Loss: 0.4830
    Test MSE: 0.4815 | Test MAE: 0.5053
    
    ==============================================================
    JANET Result
    ==============================================================
    Epoch [1/5], Loss: 0.5255
    Epoch [2/5], Loss: 0.4818
    Epoch [3/5], Loss: 0.4717
    Epoch [4/5], Loss: 0.4660
    Epoch [5/5], Loss: 0.4620
    Test MSE: 0.4770 | Test MAE: 0.4999

    ==============================================================
    IndRNN Result
    ==============================================================
    Epoch [1/5], Loss: 10.2088
    Epoch [2/5], Loss: 3.1334
    Epoch [3/5], Loss: 1.8650
    Epoch [4/5], Loss: 1.3178
    Epoch [5/5], Loss: 1.0319
    Test MSE: 0.5860 | Test MAE: 0.5792

    ==============================================================
    RAN Result
    ==============================================================
    Epoch [1/5], Loss: 0.9421
    Epoch [2/5], Loss: 0.5385
    Epoch [3/5], Loss: 0.5260
    Epoch [4/5], Loss: 0.5198
    Epoch [5/5], Loss: 0.5156
    Test MSE: 0.5010 | Test MAE: 0.5203

    ==============================================================
    SCRN Result
    ==============================================================
    Epoch [1/5], Loss: 0.5601
    Epoch [2/5], Loss: 0.4971
    Epoch [3/5], Loss: 0.4878
    Epoch [4/5], Loss: 0.4828
    Epoch [5/5], Loss: 0.4796
    Test MSE: 0.4812 | Test MAE: 0.5067

    ==============================================================
    YamRNN Result
    ==============================================================
    Epoch [1/5], Loss: 32145081035431505559552.0000
    Epoch [2/5], Loss: 32764594562860174016512.0000
    Epoch [3/5], Loss: 32549788067293997039616.0000
    Epoch [4/5], Loss: 32183375701538531966976.0000
    Epoch [5/5], Loss: 32716812923459474554880.0000
    Test MSE: 76640853335507361333248.0000 | Test MAE: 25487308800.0000
    """
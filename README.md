# StockPricePredictionDL

# 📌 Stock Price Prediction Models
This repository contains implementations of Deep Learning models to predict stock prices using a minute-by-minute price dataset. Three different models: LSTM, BiLSTM, and WaveNet are implemented. The dataset features OHLCV (Open, High, Low, Close, Volume) data along with a sentiment variable, used to analyze the polarity of news, and also includes the day of the week and the minute of the day.

## 📂 Repository Contents
- `BiLSTM.py`: Implementation of Bidirectional Long Short-Term Memory model.
- `LSTM.py`: Implementation of Long Short-Term Memory model.
- `WaveNetStock.py`: Implementation of WaveNet model.
- `TSLA_dataset_final.rar`: The minute-by-minute dataset used for training the models, containing OHLCV, date related data, and sentiment data.


## 📊 Dataset
The dataset used is `TSLA_dataset_final.rar`, a minute-by-minute price dataset featuring OHLCV and sentiment analysis data, along with the day of the week and the minute of the day, for Tesla, Inc.

## 🛠️ Models
The three models provided in this repository are LSTM, BiLSTM, and WaveNet. Each model is utilized to predict stock prices. For every model, the dataset is divided as follows: 70% for training, 15% for validation, and 15% for testing. All models employ Mean Squared Error as the loss function and use the Adam optimizer.
- **LSTM** (`LSTM.py`): Utilizes a Long Short-Term Memory network to predict stock prices. With the option to add as many LSTM layers as intended. 

- **BiLSTM** (`BiLSTM.py`): Uses a Bidirectional Long Short-Term Memory network for predicting stock prices, addressing both past and future sequences.

- **WaveNet** (`WaveNetStock.py`): Implements a WaveNet architecture for predicting stock prices. It processes sequences to capture temporal patterns and relationships in the dataset. 

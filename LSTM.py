import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredError, \
    MeanAbsoluteError
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Constants
DATA_PATH = 'TSLA_dataset_final.csv'
PRED_HORIZON = 60
N_STEPS = 120



def load_and_preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path, parse_dates=['date_time'], index_col='date_time')

    # Add new columns
    data['day_of_week'] = data.index.dayofweek.astype('int32')
    data['time'] = ((data.index.hour - 14) * 60 + data.index.minute - 30).astype('int32')
    data = data.loc['2020-05-01':]

    # Create label
    label = data['close'].shift(-PRED_HORIZON).dropna().to_frame()
    data = data.iloc[:-PRED_HORIZON]

    # Split the data in 70 train, 15 validation, 15 test
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.15, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17647, shuffle=False)

    # Scale the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_1 = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = scaler_1.fit_transform(y_train)
    y_val_scaled = scaler_1.transform(y_val)
    y_test_scaled = scaler_1.transform(y_test)

    # Transform the data for LSTM
    X_train_seq = [X_train_scaled[i - N_STEPS:i, :] for i in range(N_STEPS, len(X_train_scaled))]
    X_val_seq = [X_val_scaled[i - N_STEPS:i, :] for i in range(N_STEPS, len(X_val_scaled))]
    X_test_seq = [X_test_scaled[i - N_STEPS:i, :] for i in range(N_STEPS, len(X_test_scaled))]

    X_train_reshaped = np.array(X_train_seq)
    X_val_reshaped = np.array(X_val_seq)
    X_test_reshaped = np.array(X_test_seq)

    y_train_seq = y_train_scaled[N_STEPS:]
    y_val_seq = y_val_scaled[N_STEPS:]
    y_test_seq = y_test_scaled[N_STEPS:]

    return X_train_reshaped, y_train_seq, X_val_reshaped, y_val_seq, X_test_reshaped, y_test_seq


# Define LSTM model
def create_lstm_model(input_shape, lstm_units, l2_regs, dropout_rates):
    model = Sequential()

    for i, (units, reg, rate) in enumerate(zip(lstm_units, l2_regs, dropout_rates)):
        if i == 0:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(reg)))
        else:
            model.add(LSTM(units, activation='tanh', return_sequences=(i != len(lstm_units) - 1), kernel_regularizer=l2(reg)))
        model.add(Dropout(rate))

    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam()
    metrics = [RootMeanSquaredError(), MeanAbsolutePercentageError(), MeanSquaredError(), MeanAbsoluteError()]
    model.compile(optimizer=optimizer, loss='mse', metrics=metrics)
    model.summary()

    return model



def plot_training_results(train_loss, val_loss):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(DATA_PATH)

# Define LSTM units and regularization for each layer
lstm_units = [85, 55, 40]
l2_regs = [0.01, 0.01, 0]
dropout_rates = [0.1, 0.1, 0]
batch = 40
epoch = 40

# Create model
model = create_lstm_model(X_train.shape[1:], lstm_units, l2_regs, dropout_rates)

# Train model
history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch, validation_data=(X_val, y_val),
                        callbacks=[early_stop])

# Plot training results
plot_training_results(history.history['loss'], history.history['val_loss'])


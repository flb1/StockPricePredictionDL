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
from tensorflow.keras.layers import Bidirectional
import matplotlib
# Constants
DATA_PATH = 'TSLA_dataset_final.csv'
PRED_HORIZON = 60
N_STEPS = 120



def load_and_preprocess_data(data_path):
    """
    Load and preprocess the dataset.
    """
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

    return X_train_reshaped, y_train_seq, X_val_reshaped, y_val_seq, X_test_reshaped, y_test_seq, scaler_1

def create_stacked_bilstm_model(n_layers, n_units, dropout_rates, input_shape, n_dense_units):
    """
    Create a stacked BiLSTM model based on the specified parameters.
    """
    model = Sequential()
    for i in range(n_layers):
        return_sequences = (i != n_layers - 1)  # Set return_sequences to True except for the last layer
        model.add(Bidirectional(LSTM(n_units[i], return_sequences=return_sequences, input_shape=input_shape,
                                     kernel_regularizer=l2(0))))
        model.add(Dropout(dropout_rates[i]))

    model.add(Dense(n_dense_units, activation='relu'))
    model.add(Dense(1))
    return model

def plot_training_results(train_loss, val_loss):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Define the parameters
batch = 45
epoch = 2
n_layers = 2
n_units = [100, 80]  # Number of units for each BiLSTM layer
dropout_rates = [0.15, 0]  # Dropout rates between layers
n_dense_units = 40
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#Load and Preprocess the data
X_train, y_train, X_val, y_val, X_test, y_test, scaler_1 = load_and_preprocess_data(DATA_PATH)
input_shape = (X_train.shape[1], X_train.shape[2])

model = create_stacked_bilstm_model(n_layers, n_units, dropout_rates, input_shape, n_dense_units)

model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError(),
                                                     MeanAbsolutePercentageError(),
                                                     MeanSquaredError(), MeanAbsoluteError()])


history = model.fit(X_train, y_train, batch_size=batch, epochs=epoch,
                    validation_data=(X_val, y_val), callbacks=[early_stop])

model.summary()

# Plot training results for BiLSTM
plot_training_results(history.history['loss'], history.history['val_loss'])

##Test set Analysis
y_pred = model.predict(X_test)
y_pred_inv = scaler_1.inverse_transform(y_pred)
y_true = scaler_1.inverse_transform(y_test)

# Calculate the metrics for scaled values
mse_scaled = mean_squared_error(y_test, y_pred)

# Calculate the metrics for non-scaled (real) values
mse = mean_squared_error(y_true, y_pred_inv)

# Find the epoch at which the best model was saved
best_epoch = early_stop.stopped_epoch - early_stop.patience
if best_epoch < 0:  # Handle the case where no early stopping occurred
    best_epoch = epoch

# Displaying the MSE/Loss in the last epoch of the training and validation for comparison
print(f"Training MSE at best epoch (epoch {best_epoch}): {history.history['mean_squared_error'][best_epoch]}")
print(f"Validation MSE at best epoch (epoch {best_epoch}): {history.history['val_mean_squared_error'][best_epoch]}")
print(f"Test MSE (non-inverted scale): {mse_scaled}")
print(f"Test MSE (real scale): {mse}")

# Plot the inverse predictions and true values
plt.figure(figsize=(20, 15))
plt.plot(y_pred_inv, label='Predicted', color='orange')
plt.plot(y_true, label='True Values', color='blue')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Predictions vs real values for BiLSTM')
plt.legend()
plt.show()





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
from keras.layers import Input, Conv1D, Multiply, Dropout, Add, Activation, LeakyReLU, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2


DATA_PATH = 'TSLA_dataset_final.csv'
PRED_HORIZON = 60
N_STEPS = 240
max_dilation_rate = N_STEPS
n_filters = 32
n_layers = 12
Kernel_size = 3
batch = 40
epoch = 50
drop_rate = 0.1
l2_regularization = 0
act = 'leaky_relu'



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


def residual_block(x, num_filters, i, kernel_size, layers_per_block, dropout_rate):
    """
    A single residual block of WaveNet architecture.

    Args:
    - x: Input tensor
    - num_filters: Number of filters for convolution layers
    - i: Index of the current residual block
    - kernel_size: Size of the convolutional kernels
    - layers_per_block: Number of layers in each block
    - dropout_rate: Dropout rate

    Returns:
    - Tuple of tensors (residual_output, skip_output)
    """

    layer_num = i % layers_per_block
    dilation_rate = 2 ** layer_num

    # Causal dilated convolution
    causal_out = Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal',
                        kernel_regularizer=l2(0.001),
                        name=f'causal_conv_{dilation_rate}_s{num_filters}_i{i}')(x)

    # Gated activation units
    tanh_out = Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='tanh',
                      name=f'dilated_conv_{dilation_rate}_tanh_s{num_filters}_i{i}')(causal_out)
    sigm_out = Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='causal', activation='sigmoid',
                      name=f'dilated_conv_{dilation_rate}_sigm_s{num_filters}_i{i}')(causal_out)
    x = Multiply(name=f'gated_activation_{i}')([tanh_out, sigm_out])

    x = Dropout(dropout_rate)(x)

    # Point-wise convolution
    res_x = Conv1D(num_filters, 1, padding='valid', name=f'residual_block_{i}')(x)
    skip_x = Conv1D(num_filters, 1, padding='valid', name=f'skip_connection_{i}')(x)

    return Add()([res_x, x]), skip_x


def wavenet_model(input_shape, num_filters, num_layers, kernel_size, activation, max_dilation_rate):
    """
    WaveNet model.

    Args:
    - input_shape: Shape of the input tensor
    - num_filters: Number of filters for convolution layers
    - num_layers: Total number of layers
    - kernel_size: Size of the convolutional kernels
    - activation: Activation function ('relu' or 'leaky_relu')
    - max_dilation_rate: Maximum dilation rate

    Returns:
    - WaveNet model
    """

    layers_per_block = int(np.log2(max_dilation_rate)) + 1
    n_blocks = num_layers // layers_per_block
    if num_layers % layers_per_block != 0:
        n_blocks += 1

    x_in = Input(shape=input_shape)
    x = Conv1D(num_filters, kernel_size, dilation_rate=1, padding='causal', name='initial_causal_conv')(x_in)

    skip_connections = []

    # Construct the blocks
    for b in range(n_blocks):
        layers_in_this_block = layers_per_block if b < n_blocks - 1 else num_layers % layers_per_block
        for i in range(layers_in_this_block):
            x, skip_out = residual_block(x, num_filters, (layers_per_block * b) + i, kernel_size,
                                         layers_per_block, dropout_rate=drop_rate)
            skip_connections.append(skip_out)

    # Process skip connections
    skip_out = Add()(skip_connections)
    skip_out = Conv1D(num_filters, 1, padding='same', activation=activation, name='skip_conv')(skip_out)
    x = Add()([x, skip_out])

    # Apply activation
    if activation == 'relu':
        x = Activation('relu')(x)
    elif activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.2)(x)

    # Final processing
    x = Conv1D(num_filters, 1, name='final_conv_1')(x)

    # Apply activation
    if activation == 'relu':
        x = Activation('relu')(x)
    elif activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, name='final_prediction')(x)

    model = Model(inputs=x_in, outputs=x)
    model.compile(optimizer='adam', loss='mse')

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

def train_model(X_train_reshaped, y_train_seq, X_val_reshaped, y_val_seq):
    """Train the WaveNet model."""
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model = wavenet_model(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
                          num_filters=n_filters, num_layers=n_layers, kernel_size=Kernel_size,
                          activation=act, max_dilation_rate=N_STEPS)

    model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError(),
                                                         MeanAbsolutePercentageError(),
                                                         MeanSquaredError(), MeanAbsoluteError()])
    history = model.fit(X_train_reshaped, y_train_seq, batch_size=batch, epochs=epoch,
                        validation_data=(X_val_reshaped, y_val_seq), callbacks=[early_stop])
    plot_training_results(history.history['loss'], history.history['val_loss'])
    return model, history

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    model, history = train_model(X_train, y_train, X_val, y_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from config import LEARNING_RATE

def build_lstm(input_shape):

    tf.keras.utils.set_random_seed(47)

    model = Sequential()

    model.add(
        LSTM(
            units=14,
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(Dense(24))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )

    return model

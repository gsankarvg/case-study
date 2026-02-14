from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from config import LEARNING_RATE, SEED

def build_cnn(input_shape):

    tf.keras.utils.set_random_seed(SEED)

    model = Sequential()

    model.add(Conv2D(4, (5,1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((4,1)))

    model.add(Conv2D(8, (5,1), activation='relu'))
    model.add(MaxPooling2D((4,1)))

    model.add(Conv2D(16, (5,1), activation='relu'))
    model.add(MaxPooling2D((4,1)))

    model.add(Conv2D(32, (5,1), activation='relu'))
    model.add(MaxPooling2D((4,1)))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(168, activation='relu'))
    model.add(Dense(24))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )

    return model

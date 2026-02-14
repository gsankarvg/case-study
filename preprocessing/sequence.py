import numpy as np
from config import INPUT_WINDOW, OUTPUT_WINDOW

def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - INPUT_WINDOW - OUTPUT_WINDOW + 1):
        X.append(data[i:i+INPUT_WINDOW])
        y.append(data[i+INPUT_WINDOW:i+INPUT_WINDOW+OUTPUT_WINDOW, 0])
    return np.array(X), np.array(y)


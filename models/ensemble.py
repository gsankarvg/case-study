# models/ensemble.py

import numpy as np

def compute_weights(nrmse_lstm, nrmse_cnn):
    """
    Compute inverse-error weights based on validation NRMSE.
    """
    inv_lstm = 1 / nrmse_lstm
    inv_cnn  = 1 / nrmse_cnn

    w1 = inv_lstm / (inv_lstm + inv_cnn)
    w2 = inv_cnn  / (inv_lstm + inv_cnn)

    return w1, w2


def ensemble_predict(lstm_model, cnn_model, X_lstm, X_cnn, w1, w2):
    """
    Weighted ensemble prediction
    """
    pred_lstm = lstm_model.predict(X_lstm)
    pred_cnn  = cnn_model.predict(X_cnn)

    return w1 * pred_lstm + w2 * pred_cnn

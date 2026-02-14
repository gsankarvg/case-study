import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred, data_min, data_max):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    nrmse = rmse / (data_max - data_min)
    nmae = mae / (data_max - data_min)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "N-RMSE": nrmse,
        "N-MAE": nmae
    }

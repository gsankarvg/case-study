# main.py

import argparse
import numpy as np
import os
import random
import tensorflow as tf

from preprocessing.loader import load_dataset
from preprocessing.splitter import split_data
from preprocessing.scaler import scale_data
from preprocessing.sequence import create_sequences
from preprocessing.feature_selector import select_features

from models.lstm import build_lstm
from models.cnn import build_cnn
from models.ensemble import ensemble_predict, compute_weights

from training.trainer import train_model
from training.saver import save_model, save_metrics

from evaluation.metrics import calculate_metrics
from evaluation.visualization import plot_predictions, plot_loss

from tensorflow.keras.models import load_model
from config import INPUT_WINDOW, SEED


# ------------------------------------------------
# Argument parser
# ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)   # lstm | cnn | ensemble
parser.add_argument("--dataset", type=str, required=True) # dayton | houston
args = parser.parse_args()


# ------------------------------------------------
# Reproducibility
# ------------------------------------------------
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)


# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
df = load_dataset(f"data/{args.dataset}.csv")
df = select_features(df, args.dataset)

train, val, test = split_data(df)
train_s, val_s, test_s, scaler = scale_data(train, val, test)

X_train, y_train = create_sequences(train_s)
X_val, y_val = create_sequences(val_s)
X_test, y_test = create_sequences(test_s)

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

target_min = df["Electricity"].min()
target_max = df["Electricity"].max()


# ------------------------------------------------
# 2. Train or Load Model
# ------------------------------------------------
if args.model == "lstm":

    input_shape = (INPUT_WINDOW, X_train.shape[2])
    model = build_lstm(input_shape)

    history = train_model(model, X_train, y_train, X_val, y_val, "lstm")

    save_model(model, f"saved_models/lstm_{args.dataset}.keras")
    plot_loss(history, f"plots/lstm_{args.dataset}_loss.png")

    y_pred_scaled = model.predict(X_test)


elif args.model == "cnn":

    # reshape for Conv2D
    X_train_cnn = X_train[..., np.newaxis]
    X_val_cnn   = X_val[..., np.newaxis]
    X_test_cnn  = X_test[..., np.newaxis]

    input_shape = (INPUT_WINDOW, X_train.shape[2], 1)

    model = build_cnn(input_shape)
    history = train_model(model, X_train_cnn, y_train, X_val_cnn, y_val, "cnn")

    save_model(model, f"saved_models/cnn_{args.dataset}.keras")
    plot_loss(history, f"plots/cnn_{args.dataset}_loss.png")

    y_pred_scaled = model.predict(X_test_cnn)


elif args.model == "ensemble":

    lstm_model = load_model(f"saved_models/lstm_{args.dataset}.keras")
    cnn_model  = load_model(f"saved_models/cnn_{args.dataset}.keras")

    X_val_cnn  = X_val[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # validation predictions
    val_pred_lstm = lstm_model.predict(X_val)
    val_pred_cnn  = cnn_model.predict(X_val_cnn)

    # inverse scaling validation
    val_pred_lstm = val_pred_lstm * (target_max - target_min) + target_min
    val_pred_cnn  = val_pred_cnn  * (target_max - target_min) + target_min
    y_val_orig    = y_val * (target_max - target_min) + target_min

    # compute validation NRMSE
    val_metrics_lstm = calculate_metrics(
        y_val_orig.flatten(),
        val_pred_lstm.flatten(),
        target_min,
        target_max
    )

    val_metrics_cnn = calculate_metrics(
        y_val_orig.flatten(),
        val_pred_cnn.flatten(),
        target_min,
        target_max
    )

    w1, w2 = compute_weights(
        val_metrics_lstm["N-RMSE"],
        val_metrics_cnn["N-RMSE"]
    )

    # ensemble prediction
    y_pred_scaled = ensemble_predict(
        lstm_model,
        cnn_model,
        X_test,
        X_test_cnn,
        w1,
        w2
    )

else:
    raise ValueError("Unsupported model type")


# ------------------------------------------------
# 3. Inverse Scaling
# ------------------------------------------------
y_pred = y_pred_scaled * (target_max - target_min) + target_min
y_true = y_test * (target_max - target_min) + target_min


# ------------------------------------------------
# 4. Evaluate
# ------------------------------------------------
metrics = calculate_metrics(
    y_true.flatten(),
    y_pred.flatten(),
    target_min,
    target_max
)

print(metrics)


# ------------------------------------------------
# 5. Save Results
# ------------------------------------------------
model_name = args.model
save_metrics(metrics, f"results/{model_name}_{args.dataset}_metrics.json")

plot_predictions(
    y_true,
    y_pred,
    f"plots/{model_name}_{args.dataset}_prediction.png"
)
